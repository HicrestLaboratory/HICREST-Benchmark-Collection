/**
 * ml_bench — CPU baseline for ML operator shapes
 *
 * Build flags
 *   BACKEND : openblas (default) | mkl | eigen
 *   DTYPE   : float (default) | double
 *
 *   make                          # OpenBLAS, float32, OpenMP on
 *   make BACKEND=mkl DTYPE=double # MKL, float64
 *   make OMP=0                    # disable OpenMP (serial baseline)
 *
 * Acceleration strategy
 *   BLAS calls are kept as-is (the library handles internal threading).
 *   Where we own the loops two complementary techniques are applied:
 *
 *   1. Fuse into a single BLAS call whenever possible
 *      - conv2d (standard, groups==1): stack all N im2col outputs into one
 *        matrix, then issue a single cblas_?gemm with CblasTrans on B.
 *      - conv2d (depthwise, groups==Cin): one cblas_?gemm_batch call across
 *        all groups (each group is a 1-channel micro-GEMM).
 *      - batched_gemm (OpenBLAS/MKL): cblas_?gemm_batch so the library can
 *        schedule all slices internally.
 *
 *   2. OpenMP thread parallelism  (#pragma omp parallel for)
 *      - im2col            : parallel over channels (c loop); fill also
 *                            parallelised per-row inside each thread
 *      - conv2d path A (N>1 de-interleave): parallel collapse(2) over C_out×N
 *      - conv2d path B (depthwise): parallel over N; inner im2col guarded by
 *                            if(groups > 64) to avoid nested overhead
 *      - conv2d path C (fallback): parallel collapse(2) over N×groups
 *      - batched_gemm (Eigen): parallel over batch slices
 *      - layer_norm        : parallel over rows (N)
 *      - softmax           : parallel over rows (N)
 *      - gelu              : parallel over elements
 *      - elementwise_add   : parallel + SIMD over elements
 *      - embedding_lookup  : parallel over sequence positions
 *
 *   3. OpenMP SIMD  (#pragma omp simd)
 *      Applied to short inner loops that are vectorisation-safe but where the
 *      compiler needs an explicit hint due to function calls or apparent aliasing:
 *      - layer_norm  : mean accumulation, variance accumulation, normalise pass
 *      - softmax     : max reduction, exp+sum pass, scale pass
 *      - elementwise_add : combined with parallel for (parallel for simd)
 */

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#if defined(_OPENMP)
  #include <omp.h>
#endif

// =============================================================================
// Data type selection
//   make DTYPE=float  (default) -> scalar_t = float,  uses ?sgemm
//   make DTYPE=double           -> scalar_t = double, uses ?dgemm
// =============================================================================

#if defined(USE_DOUBLE)
  using scalar_t = double;
  #define DTYPE_NAME "float64"
#else
  using scalar_t = float;
  #define DTYPE_NAME "float32"
#endif

// =============================================================================
// Backend abstraction
//   Only this section + gemm() + batched_gemm() change when you swap backends.
// =============================================================================

#if defined(USE_MKL)
  #include <mkl_cblas.h>
  #define BACKEND_NAME "Intel MKL"

#elif defined(USE_EIGEN)
  #include <Eigen/Dense>
  #define BACKEND_NAME "Eigen"

#else  // default: OpenBLAS
  #include <cblas.h>
  #define BACKEND_NAME "OpenBLAS"
#endif

// =============================================================================
// BLAS dispatch: one set of inline wrappers per scalar type so the rest of the
// code is type-generic via scalar_t.
// =============================================================================

// ---- single GEMM ------------------------------------------------------------

static inline void blas_gemm(int M, int N, int K,
                              scalar_t alpha,
                              const scalar_t* A, int lda,
                              const scalar_t* B, int ldb,
                              scalar_t beta,
                              scalar_t*       C, int ldc) {
#if defined(USE_EIGEN)
    using namespace Eigen;
    Map<const Matrix<scalar_t, Dynamic, Dynamic, RowMajor>> eA(A, M, K);
    Map<const Matrix<scalar_t, Dynamic, Dynamic, RowMajor>> eB(B, K, N);
    Map<Matrix<scalar_t, Dynamic, Dynamic, RowMajor>>       eC(C, M, N);
    eC = alpha * (eA * eB) + beta * eC;
    (void)lda; (void)ldb; (void)ldc;
#elif defined(USE_DOUBLE)
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
#else
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
}

// Convenience: square leading dimensions (row-major, no padding)
static inline void gemm(int M, int K, int N,
                        const scalar_t* A, const scalar_t* B, scalar_t* C,
                        scalar_t alpha = scalar_t(1), scalar_t beta = scalar_t(0)) {
    blas_gemm(M, N, K, alpha, A, K, B, N, beta, C, N);
}

// ---- batched GEMM -----------------------------------------------------------
// Caller provides flat arrays [batch, M, K], [batch, K, N], [batch, M, N].
//
// MKL    : cblas_?gemm_batch  — native batched kernel, best scheduling
// OpenBLAS: OpenMP loop over cblas_?gemm  — cblas_?gemm_batch exists in
//           OpenBLAS >= 0.3.x but is not present in all distro packages;
//           a parallel loop is equivalent and always links correctly
// Eigen  : OpenMP loop over Eigen products

static void batched_gemm(int batch, int M, int K, int N,
                         const scalar_t* A,
                         const scalar_t* B,
                         scalar_t*       C) {
#if defined(USE_EIGEN)
    using namespace Eigen;
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < batch; ++b) {
        Map<const Matrix<scalar_t, Dynamic, Dynamic, RowMajor>> eA(A + b*M*K, M, K);
        Map<const Matrix<scalar_t, Dynamic, Dynamic, RowMajor>> eB(B + b*K*N, K, N);
        Map<Matrix<scalar_t, Dynamic, Dynamic, RowMajor>>       eC(C + b*M*N, M, N);
        eC.noalias() = eA * eB;
    }

#elif defined(USE_MKL)
    // MKL provides a native batched kernel — hand it all slices at once so it
    // can optimise scheduling and cache reuse across the batch.
    std::vector<const scalar_t*> pa(batch), pb(batch);
    std::vector<scalar_t*>       pc(batch);
    for (int b = 0; b < batch; ++b) {
        pa[b] = A + b * M * K;
        pb[b] = B + b * K * N;
        pc[b] = C + b * M * N;
    }
    const scalar_t alpha = scalar_t(1), beta = scalar_t(0);
    const CBLAS_TRANSPOSE nt = CblasNoTrans;
    const int grp = 1;
  #if defined(USE_DOUBLE)
    cblas_dgemm_batch(CblasRowMajor, &nt, &nt,
                      &M, &N, &K,
                      &alpha, pa.data(), &K,
                              pb.data(), &N,
                      &beta,  pc.data(), &N,
                      grp, &batch);
  #else
    cblas_sgemm_batch(CblasRowMajor, &nt, &nt,
                      &M, &N, &K,
                      &alpha, pa.data(), &K,
                              pb.data(), &N,
                      &beta,  pc.data(), &N,
                      grp, &batch);
  #endif

#else  // OpenBLAS: parallelise the loop; each slice is independent
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < batch; ++b) {
  #if defined(USE_DOUBLE)
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K, 1.0,
                    A + b * M * K, K,
                    B + b * K * N, N,
                    0.0,
                    C + b * M * N, N);
  #else
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K, 1.f,
                    A + b * M * K, K,
                    B + b * K * N, N,
                    0.f,
                    C + b * M * N, N);
  #endif
    }
#endif
}

// =============================================================================
// im2col
//   Input  : [C, H, W]  (one sample / one group)
//   Output : [C*kH*kW,  H_out*W_out]
//
//   The channel loop is independent -> parallelise with OpenMP.
// =============================================================================

static void im2col(const scalar_t* src,
                   int C, int H, int W,
                   int kH, int kW,
                   int pH, int pW,
                   int sH, int sW,
                   scalar_t* dst) {
    const int H_out = (H + 2 * pH - kH) / sH + 1;
    const int W_out = (W + 2 * pW - kW) / sW + 1;
    const int col_w = H_out * W_out;

    // Each (c, kh, kw) triple writes to a distinct contiguous row in dst ->
    // fully independent.  We parallelise the outermost c loop and zero-init
    // inside each thread's slice so the fill is also parallel.
    #pragma omp parallel for schedule(static) if(C * kH * kW > 64)
    for (int c = 0; c < C; ++c) {
        for (int kh = 0; kh < kH; ++kh) {
            for (int kw = 0; kw < kW; ++kw) {
                const int row = (c * kH + kh) * kW + kw;
                scalar_t* dst_row = dst + row * col_w;

                // Zero the whole output row first, then fill valid positions.
                // This is faster than a branch inside the hot inner loop when
                // padding is non-zero, and free when padding is zero.
                std::fill(dst_row, dst_row + col_w, scalar_t(0));

                #pragma omp simd
                for (int oh = 0; oh < H_out; ++oh) {
                    for (int ow = 0; ow < W_out; ++ow) {
                        const int ih = oh * sH - pH + kh;
                        const int iw = ow * sW - pW + kw;
                        if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                            dst_row[oh * W_out + ow] =
                                src[(c * H + ih) * W + iw];
                    }
                }
            }
        }
    }
}

// =============================================================================
// Conv2D via im2col + GEMM
//
//   Two specialised paths, both keep BLAS hot:
//
//   A) Standard conv (groups == 1)
//      For a batch of N images we run im2col on each sample then stack the
//      resulting col matrices into one big matrix:
//        col_all  [N * H_out*W_out,  Cig*kH*kW]
//        weight   [C_out,            Cig*kH*kW]
//      -> one GEMM: output [C_out, N*H_out*W_out]
//      This keeps the BLAS kernel busy longer and avoids N round-trips through
//      the cache for the weight tensor.
//      Note: for N=1 (our benchmark) this is identical to the original path
//      but generalises cleanly.
//
//   B) Depthwise conv (groups == C_in, Cig == 1)
//      Each group g is independently:
//        col_g   [kH*kW,  H_out*W_out]   (one input channel patch matrix)
//        weight_g [1,      kH*kW]         (one output filter)
//      -> cblas_?gemm_batch over all groups in one BLAS call.
//
//   C) Arbitrary groups fallback
//      Serial im2col + GEMM per (n, g), same as before but with OpenMP over n.
// =============================================================================

static void conv2d(const scalar_t* input, const scalar_t* weight, scalar_t* output,
                   int N, int C_in, int H, int W,
                   int C_out, int kH, int kW,
                   int groups,
                   int pH = 0, int pW = 0,
                   int sH = 1, int sW = 1) {
    assert(C_in  % groups == 0);
    assert(C_out % groups == 0);

    const int Cig   = C_in  / groups;
    const int Cog   = C_out / groups;
    const int H_out = (H + 2 * pH - kH) / sH + 1;
    const int W_out = (W + 2 * pW - kW) / sW + 1;
    const int col_rows = Cig * kH * kW;
    const int col_cols = H_out * W_out;

    // ── Path A: standard convolution (groups == 1) ───────────────────────────
    if (groups == 1) {
        // col_all: [N * col_cols,  col_rows]  (stacked im2col outputs)
        std::vector<scalar_t> col_all(N * col_cols * col_rows);

        // im2col each sample in parallel (outer loop); im2col itself also has
        // an OpenMP pragma for its inner c-loop, but at groups==1 both levels
        // are active. The inner pragma has an `if` guard so small C_in won't
        // spawn extra teams.
        #pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            im2col(input + n * C_in * H * W,
                   Cig, H, W, kH, kW, pH, pW, sH, sW,
                   col_all.data() + n * col_cols * col_rows);
        }

        // weight : [C_out, col_rows]
        // col_all: [N*col_cols, col_rows]  -> treat as [col_rows, N*col_cols]^T
        // We want output [C_out, N*col_cols] then reshape to [N, C_out, Ho, Wo].
        // With row-major layout this is:
        //   C = weight * col_all^T  but simpler to call with col_all transposed:
        //   output[C_out, N*col_cols] = weight[C_out, col_rows] * col_all^T[col_rows, N*col_cols]
        // col_all is stored as [N*col_cols, col_rows] (row-major), so col_all^T
        // is accessed by passing CblasTrans.
        // Rather than add a transpose flag to our thin wrapper, just call
        // blas_gemm with the explicit leading-dimension control:
        //   A = weight  [C_out, col_rows],  lda = col_rows
        //   B = col_all [N*col_cols, col_rows], ldb = col_rows, but B^T needed:
        //     -> swap M/N and use CblasTrans on B
        // Easiest: reuse gemm() which does NoTrans/NoTrans. col_all is already
        // [N*col_cols, col_rows] which we treat as B in the product
        //   output = weight  x  col_all^T
        //          = [C_out, col_rows] x [col_rows, N*col_cols]
        // so call cblas with B = col_all and CblasTrans for B:
#if defined(USE_EIGEN)
        {
            using namespace Eigen;
            Map<const Matrix<scalar_t, Dynamic, Dynamic, RowMajor>>
                W_(weight,   C_out,         col_rows);
            Map<const Matrix<scalar_t, Dynamic, Dynamic, RowMajor>>
                CA(col_all.data(), N * col_cols, col_rows);
            Map<Matrix<scalar_t, Dynamic, Dynamic, RowMajor>>
                Out(output, C_out, N * col_cols);
            Out.noalias() = W_ * CA.transpose();
        }
#elif defined(USE_DOUBLE)
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    C_out, N * col_cols, col_rows,
                    1.0, weight, col_rows,
                         col_all.data(), col_rows,
                    0.0, output, N * col_cols);
#else
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    C_out, N * col_cols, col_rows,
                    1.f, weight, col_rows,
                         col_all.data(), col_rows,
                    0.f, output, N * col_cols);
#endif
        // The GEMM writes output as [C_out, N*col_cols].
        // We need [N, C_out, H_out, W_out].  For N==1 no reorder needed.
        if (N > 1) {
            std::vector<scalar_t> tmp(output, output + N * C_out * H_out * W_out);
            // tmp[c, n*col_cols + p] -> output[n, c, p]
            #pragma omp parallel for schedule(static) collapse(2)
            for (int c = 0; c < C_out; ++c)
                for (int n = 0; n < N; ++n)
                    std::memcpy(output + (n * C_out + c) * col_cols,
                                tmp.data() + c * (N * col_cols) + n * col_cols,
                                col_cols * sizeof(scalar_t));
        }
        return;
    }

    // ── Path B: depthwise conv (groups == C_in, Cig == 1) ────────────────────
    // Each group has: weight_g [Cog=1, kH*kW], col_g [kH*kW, H_out*W_out]
    // All groups processed in one batched GEMM call per sample.
    if (groups == C_in && Cig == 1) {
        const int col_size = kH * kW * col_cols;  // per-group col buffer

        // Parallelise over the batch dimension.  Each thread needs its own
        // col_all scratch buffer (one buffer per sample is fine since all
        // groups within a sample are processed serially by batched_gemm).
        #pragma omp parallel
        {
            std::vector<scalar_t> col_all(groups * col_size);

            #pragma omp for schedule(static)
            for (int n = 0; n < N; ++n) {
                // im2col all input channels — inner parallel for is guarded by
                // the if() clause, so no nested-parallel overhead for tiny C.
                #pragma omp parallel for schedule(static) if(groups > 64)
                for (int g = 0; g < groups; ++g) {
                    im2col(input + (n * C_in + g) * H * W,
                           1, H, W, kH, kW, pH, pW, sH, sW,
                           col_all.data() + g * col_size);
                }

                scalar_t* out_n = output + n * C_out * col_cols;
                batched_gemm(groups,
                             /*M=*/Cog, /*K=*/kH * kW, /*N=*/col_cols,
                             weight, col_all.data(), out_n);
            }
        }
        return;
    }

    // ── Path C: arbitrary groups fallback ────────────────────────────────────
    // Parallel over the batch dimension; each thread gets its own col buffer.
    #pragma omp parallel
    {
        std::vector<scalar_t> col(col_rows * col_cols);
        #pragma omp for schedule(static) collapse(2)
        for (int n = 0; n < N; ++n) {
            for (int g = 0; g < groups; ++g) {
                const scalar_t* in_ptr  = input  + (n * C_in  + g * Cig) * H * W;
                const scalar_t* wt_ptr  = weight + g * Cog * col_rows;
                scalar_t*       out_ptr = output + (n * C_out + g * Cog) * col_cols;
                im2col(in_ptr, Cig, H, W, kH, kW, pH, pW, sH, sW, col.data());
                gemm(Cog, col_rows, col_cols, wt_ptr, col.data(), out_ptr);
            }
        }
    }
}

// =============================================================================
// LayerNorm  --  y = (x - mean) / sqrt(var + eps) * gamma + beta
//   Input  : [N, D]  (rows are independent -> parallelise over N)
// =============================================================================

static void layer_norm(const scalar_t* x, scalar_t* y,
                       const scalar_t* gamma, const scalar_t* beta,
                       int N, int D, scalar_t eps = scalar_t(1e-5)) {
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; ++n) {
        const scalar_t* row = x + n * D;
        scalar_t*       out = y + n * D;

        scalar_t mean = scalar_t(0);
        #pragma omp simd reduction(+:mean)
        for (int d = 0; d < D; ++d) mean += row[d];
        mean /= static_cast<scalar_t>(D);

        scalar_t var = scalar_t(0);
        #pragma omp simd reduction(+:var)
        for (int d = 0; d < D; ++d) { scalar_t diff = row[d] - mean; var += diff * diff; }
        var /= static_cast<scalar_t>(D);
        const scalar_t inv_std = scalar_t(1) / std::sqrt(var + eps);

        #pragma omp simd
        for (int d = 0; d < D; ++d)
            out[d] = (row[d] - mean) * inv_std * gamma[d] + beta[d];
    }
}

// =============================================================================
// Softmax  --  stable, row-wise (rows are independent -> parallelise over N)
// =============================================================================

static void softmax(const scalar_t* x, scalar_t* y, int N, int D) {
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; ++n) {
        const scalar_t* row = x + n * D;
        scalar_t*       out = y + n * D;

        // max reduction — scalar, not SIMD-reducible with standard OpenMP
        scalar_t mx = row[0];
        #pragma omp simd reduction(max:mx)
        for (int d = 1; d < D; ++d) if (row[d] > mx) mx = row[d];

        scalar_t sum = scalar_t(0);
        #pragma omp simd reduction(+:sum)
        for (int d = 0; d < D; ++d) { out[d] = std::exp(row[d] - mx); sum += out[d]; }

        const scalar_t inv_sum = scalar_t(1) / sum;
        #pragma omp simd
        for (int d = 0; d < D; ++d) out[d] *= inv_sum;
    }
}

// =============================================================================
// GELU  --  tanh approximation, element-wise
// =============================================================================

static constexpr scalar_t kGelu_c0 = scalar_t(0.7978845608);  // sqrt(2/pi)
static constexpr scalar_t kGelu_c1 = scalar_t(0.044715);

static void gelu(const scalar_t* x, scalar_t* y, int N) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        const scalar_t v = x[i];
        y[i] = scalar_t(0.5) * v *
               (scalar_t(1) + std::tanh(kGelu_c0 * (v + kGelu_c1 * v * v * v)));
    }
}

// =============================================================================
// Elementwise add  --  z = x + y
// =============================================================================

static void elementwise_add(const scalar_t* x, const scalar_t* y, scalar_t* z, int N) {
    #pragma omp parallel for simd schedule(static)
    for (int i = 0; i < N; ++i) z[i] = x[i] + y[i];
}

// =============================================================================
// Embedding lookup  --  out[i] = table[indices[i]]
//   Random gather: no data dependency between rows -> parallelise over i.
// =============================================================================

static void embedding_lookup(const scalar_t* table, const int* indices, scalar_t* out,
                              int seq_len, int embed_dim) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < seq_len; ++i)
        std::memcpy(out   + i          * embed_dim,
                    table + indices[i] * embed_dim,
                    embed_dim * sizeof(scalar_t));
}

// =============================================================================
// Benchmark harness
// =============================================================================

struct BenchResult {
    std::string name;
    std::string section;    // set by print_row() from g_current_section
    std::string category;   // "compute" | "memory"
    double mean_ms;
    double min_ms;
    double max_ms;
    double gflops;
    double gbytes_per_s;
};

static BenchResult run_bench(const std::string& name,
                             const std::string& category,
                             std::function<void()> fn,
                             double flop_count,
                             double byte_count,
                             int warmup = 3,
                             int iters  = 10) {
    for (int i = 0; i < warmup; ++i) fn();

    std::vector<double> times(iters);
    for (int i = 0; i < iters; ++i) {
        double t0 = std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        fn();
        times[i] = std::chrono::duration<double, std::milli>(
                       std::chrono::high_resolution_clock::now().time_since_epoch()).count() - t0;
    }

    double mean = std::accumulate(times.begin(), times.end(), 0.0) / iters;
    double mn   = *std::min_element(times.begin(), times.end());
    double mx   = *std::max_element(times.begin(), times.end());
    double gf   = (flop_count > 0) ? flop_count / (mean * 1e6) : 0.0;
    double gb   = (byte_count  > 0) ? byte_count  / (mean * 1e6) : 0.0;

    return {name, /*section=*/"", category, mean, mn, mx, gf, gb};
}

// =============================================================================
// Output — table and JSON
// =============================================================================

// Global flag set by --json / -j at startup.
// Kept in one place so every print function can read it without extra args.
static bool g_json = false;

// ---- JSON helpers -----------------------------------------------------------

// Escape the handful of characters that are special inside a JSON string.
// We only need to handle what appears in our benchmark names (no Unicode, no
// control chars other than the occasional backslash from Windows paths).
static std::string json_str(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    out += '"';
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;
        }
    }
    out += '"';
    return out;
}

static std::string json_dbl(double v, int prec = 4) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(prec) << v;
    return ss.str();
}

// ---- Table helpers ----------------------------------------------------------

static constexpr int COL_NAME = 52;
static constexpr int COL_W    = 10;
static constexpr int COL_RATE = 13;
static constexpr int TOTAL_W  = COL_NAME + COL_W*3 + COL_RATE*2;

static std::string hline() { return std::string(TOTAL_W, '-'); }

// ---- Unified print functions ------------------------------------------------
// In JSON mode these are all no-ops; output is deferred to print_json_output().

static std::string g_current_section;  // tracked for JSON grouping

static void print_section(const std::string& title) {
    g_current_section = title;
    if (!g_json)
        std::cout << "\n[" << title << "]\n";
}

static void print_header() {
    if (g_json) return;  // JSON header is emitted once at the end
    std::cout << "\nBackend : " << BACKEND_NAME
              << "   dtype=" << DTYPE_NAME;
#if defined(_OPENMP)
    std::cout << "   threads=" << omp_get_max_threads();
#else
    std::cout << "   OpenMP=off";
#endif
    std::cout << "\n" << hline() << "\n";
    std::cout << std::left  << std::setw(COL_NAME) << "Benchmark"
              << std::right << std::setw(COL_W)    << "Mean(ms)"
              << std::setw(COL_W)                  << "Min(ms)"
              << std::setw(COL_W)                  << "Max(ms)"
              << std::setw(COL_RATE)               << "GFLOP/s"
              << std::setw(COL_RATE)               << "GB/s"
              << "\n" << hline() << "\n";
}

// BenchResult is extended with section at record time so JSON can group by it.
// We store it on the result itself rather than re-deriving it from a parallel list.
static void print_row(BenchResult& r) {
    // Stamp the section onto the result for later JSON grouping.
    r.section = g_current_section;

    if (g_json) return;  // table row suppressed; entire table printed at end

    std::cout << std::left  << std::setw(COL_NAME) << r.name
              << std::right << std::fixed << std::setprecision(3)
              << std::setw(COL_W) << r.mean_ms
              << std::setw(COL_W) << r.min_ms
              << std::setw(COL_W) << r.max_ms
              << std::setprecision(2)
              << std::setw(COL_RATE) << r.gflops
              << std::setw(COL_RATE) << r.gbytes_per_s
              << "\n";
}

// ---- JSON output (emitted once, after all benchmarks finish) ----------------

static void print_json_output(const std::vector<BenchResult>& results) {
    // Compute summary totals (same logic as the table summary block).
    double compute_ms = 0, memory_ms = 0, compute_gf = 0, peak_gb = 0;
    for (auto& r : results) {
        if (r.category == "compute") { compute_ms += r.mean_ms; compute_gf += r.gflops; }
        else                         { memory_ms  += r.mean_ms; peak_gb = std::max(peak_gb, r.gbytes_per_s); }
    }

    int omp_threads = 1;
#if defined(_OPENMP)
    omp_threads = omp_get_max_threads();
#endif

    std::ostringstream o;
    o << "{\n";

    // ── metadata ─────────────────────────────────────────────────────────────
    o << "  \"metadata\": {\n";
    o << "    \"backend\": "   << json_str(BACKEND_NAME) << ",\n";
    o << "    \"dtype\": "     << json_str(DTYPE_NAME)   << ",\n";
    o << "    \"omp_threads\": " << omp_threads           << ",\n";
    o << "    \"warmup_iters\": 3,\n";
    o << "    \"bench_iters\": 10\n";
    o << "  },\n";

    // ── results array ────────────────────────────────────────────────────────
    o << "  \"results\": [\n";
    for (std::size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        o << "    {\n";
        o << "      \"name\": "     << json_str(r.name)     << ",\n";
        o << "      \"section\": "  << json_str(r.section)  << ",\n";
        o << "      \"category\": " << json_str(r.category) << ",\n";
        o << "      \"mean_ms\": "  << json_dbl(r.mean_ms)  << ",\n";
        o << "      \"min_ms\": "   << json_dbl(r.min_ms)   << ",\n";
        o << "      \"max_ms\": "   << json_dbl(r.max_ms)   << ",\n";
        o << "      \"gflops\": "   << json_dbl(r.gflops)   << ",\n";
        o << "      \"gbytes_per_s\": " << json_dbl(r.gbytes_per_s) << "\n";
        o << "    }" << (i + 1 < results.size() ? "," : "") << "\n";
    }
    o << "  ],\n";

    // ── summary ──────────────────────────────────────────────────────────────
    o << "  \"summary\": {\n";
    o << "    \"total_operators\": " << results.size() << ",\n";
    o << "    \"compute_total_ms\": "  << json_dbl(compute_ms) << ",\n";
    o << "    \"compute_gflops_sum\": " << json_dbl(compute_gf) << ",\n";
    o << "    \"memory_total_ms\": "   << json_dbl(memory_ms)  << ",\n";
    o << "    \"peak_gbytes_per_s\": " << json_dbl(peak_gb)    << ",\n";
    o << "    \"grand_total_ms\": "    << json_dbl(compute_ms + memory_ms) << "\n";
    o << "  }\n";

    o << "}\n";
    std::cout << o.str();
}

// =============================================================================
// Helpers
// =============================================================================

static void fill_random(std::vector<scalar_t>& v,
                        scalar_t lo = scalar_t(-0.1),
                        scalar_t hi = scalar_t( 0.1)) {
    static std::mt19937 rng(42);
    std::uniform_real_distribution<scalar_t> dist(lo, hi);
    for (auto& x : v) x = dist(rng);
}

static void fill_int(std::vector<int>& v, int lo, int hi) {
    static std::mt19937 rng(99);
    std::uniform_int_distribution<int> dist(lo, hi);
    for (auto& x : v) x = dist(rng);
}

// =============================================================================
// Per-operator benchmark helpers
// =============================================================================

static BenchResult bench_gemm(const std::string& lbl, int M, int K, int N) {
    std::vector<scalar_t> A(M*K), B(K*N), C(M*N, scalar_t(0));
    fill_random(A); fill_random(B);
    return run_bench(lbl, "compute",
        [&]{ gemm(M, K, N, A.data(), B.data(), C.data()); },
        2.0 * M * K * N, 0);
}

static BenchResult bench_batched_gemm(const std::string& lbl,
                                      int batch, int M, int K, int N) {
    std::vector<scalar_t> A(batch*M*K), B(batch*K*N), C(batch*M*N, scalar_t(0));
    fill_random(A); fill_random(B);
    return run_bench(lbl, "compute",
        [&]{ batched_gemm(batch, M, K, N, A.data(), B.data(), C.data()); },
        2.0 * batch * M * K * N, 0);
}

static BenchResult bench_conv(const std::string& lbl,
                              int N, int C_in, int H, int W,
                              int C_out, int kH, int kW, int groups,
                              int pH = 0, int pW = 0,
                              int sH = 1, int sW = 1) {
    const int H_out = (H + 2*pH - kH)/sH + 1;
    const int W_out = (W + 2*pW - kW)/sW + 1;
    std::vector<scalar_t> input (N * C_in  * H * W);
    std::vector<scalar_t> weight(C_out * (C_in/groups) * kH * kW);
    std::vector<scalar_t> output(N * C_out * H_out * W_out, scalar_t(0));
    fill_random(input); fill_random(weight);
    double flops = 2.0 * N * C_out * H_out * W_out * (C_in/groups) * kH * kW;
    return run_bench(lbl, "compute",
        [&]{ conv2d(input.data(), weight.data(), output.data(),
                    N, C_in, H, W, C_out, kH, kW, groups, pH, pW, sH, sW); },
        flops, 0);
}

static BenchResult bench_layer_norm(const std::string& lbl, int N, int D) {
    std::vector<scalar_t> x(N*D), y(N*D), gamma(D, scalar_t(1)), beta(D, scalar_t(0));
    fill_random(x);
    double bytes = sizeof(scalar_t) * (2.0*N*D + 2.0*D);
    return run_bench(lbl, "memory",
        [&]{ layer_norm(x.data(), y.data(), gamma.data(), beta.data(), N, D); },
        7.0 * N * D, bytes);
}

static BenchResult bench_softmax(const std::string& lbl, int N, int D) {
    std::vector<scalar_t> x(N*D), y(N*D);
    fill_random(x);
    double bytes = sizeof(scalar_t) * 2.0 * N * D;
    return run_bench(lbl, "memory",
        [&]{ softmax(x.data(), y.data(), N, D); },
        static_cast<double>(N) * D * 5, bytes);
}

static BenchResult bench_gelu(const std::string& lbl, int N) {
    std::vector<scalar_t> x(N), y(N);
    fill_random(x);
    double bytes = sizeof(scalar_t) * 2.0 * N;
    return run_bench(lbl, "memory",
        [&]{ gelu(x.data(), y.data(), N); },
        static_cast<double>(N) * 8, bytes);
}

static BenchResult bench_add(const std::string& lbl, int N) {
    std::vector<scalar_t> x(N), y(N), z(N);
    fill_random(x); fill_random(y);
    double bytes = sizeof(scalar_t) * 3.0 * N;
    return run_bench(lbl, "memory",
        [&]{ elementwise_add(x.data(), y.data(), z.data(), N); },
        static_cast<double>(N), bytes);
}

static BenchResult bench_embedding(const std::string& lbl,
                                   int seq_len, int vocab_size, int embed_dim) {
    std::vector<scalar_t> table(vocab_size * embed_dim);
    std::vector<scalar_t> out  (seq_len    * embed_dim);
    std::vector<int>      idx  (seq_len);
    fill_random(table);
    fill_int(idx, 0, vocab_size - 1);
    double bytes = sizeof(scalar_t) * 2.0 * seq_len * embed_dim;
    return run_bench(lbl, "memory",
        [&]{ embedding_lookup(table.data(), idx.data(), out.data(),
                              seq_len, embed_dim); },
        0, bytes);
}

// =============================================================================
// main
// =============================================================================

int main(int argc, char* argv[]) {
    // ── CLI flags ─────────────────────────────────────────────────────────────
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--json" || arg == "-j") {
            g_json = true;
        } else {
            std::cerr << "Usage: " << argv[0] << " [--json|-j]\n"
                      << "  --json, -j   Emit results as a single JSON document\n";
            return 1;
        }
    }

    print_header();

    std::vector<BenchResult> results;
    // run() takes by value; print_row() stamps section onto r before we store it.
    auto run = [&](BenchResult r) { print_row(r); results.push_back(std::move(r)); };

    // ── DINOv2 ViT-B ─────────────────────────────────────────────────────────
    print_section("DINOv2 ViT-B -- Compute-bound");

    run(bench_conv("DINOv2 patch_embed [1,3,224,224]x[768,3,16,16]",
                   1, 3, 224, 224, 768, 16, 16, 1, 0, 0, 16, 16));

    for (int i = 1; i <= 12; ++i)
        run(bench_gemm("DINOv2 mm blk" + std::to_string(i) + " [201,768]x[768,768]",
                       201, 768, 768));

    print_section("DINOv2 ViT-B -- Batched GEMM (multi-head attention)");
    run(bench_batched_gemm("DINOv2 bmm QK^T  [12,201,64]x[12,64,201]",  12, 201,  64, 201));
    run(bench_batched_gemm("DINOv2 bmm AV    [12,201,201]x[12,201,64]", 12, 201, 201,  64));

    // ── ConvNeXt-B ───────────────────────────────────────────────────────────
    print_section("ConvNeXt-B -- Compute-bound");

    struct CS { int N,Ci,H,W,Co,kH,kW,g; const char* lbl; };
    CS cnb[] = {
        {1,    3,224,224,  128, 4,4,   1,"ConvNeXt stem          [1,3,224,224]x[128,3,4,4]"   },
        {1,  128, 56, 56,  128, 7,7, 128,"ConvNeXt dw-s1 128     [1,128,56,56]x[128,1,7,7]"  },
        {1,  128, 56, 56,  256, 2,2,   1,"ConvNeXt down 128->256 [1,128,56,56]x[256,128,2,2]"},
        {1,  256, 28, 28,  256, 7,7, 256,"ConvNeXt dw-s2 256     [1,256,28,28]x[256,1,7,7]"  },
        {1,  256, 28, 28,  512, 2,2,   1,"ConvNeXt down 256->512 [1,256,28,28]x[512,256,2,2]"},
        {1,  512, 14, 14,  512, 7,7, 512,"ConvNeXt dw-s3 512     [1,512,14,14]x[512,1,7,7]"  },
        {1,  512, 14, 14, 1024, 2,2,   1,"ConvNeXt down 512->1k  [1,512,14,14]x[1024,512,2,2]"},
        {1, 1024,  7,  7, 1024, 7,7,1024,"ConvNeXt dw-s4 1024    [1,1024,7,7]x[1024,1,7,7]"  },
    };
    for (auto& s : cnb) {
        bool dw   = s.g > 1;
        bool stem = s.kH == 4;
        int pH = dw ? 3 : 0, pW = dw ? 3 : 0;
        int sH = (!dw && !stem) ? 2 : (stem ? 4 : 1), sW = sH;
        run(bench_conv(s.lbl, s.N,s.Ci,s.H,s.W,s.Co,s.kH,s.kW,s.g,pH,pW,sH,sW));
    }

    print_section("Pointwise 1x1 Conv -- ConvNeXt MLP expansion");
    run(bench_conv("PW 256->1024  [1,256,28,28]x[1024,256,1,1]",   1,  256,28,28, 1024,1,1,1));
    run(bench_conv("PW 1024->256  [1,1024,28,28]x[256,1024,1,1]",  1, 1024,28,28,  256,1,1,1));
    run(bench_conv("PW 512->2048  [1,512,14,14]x[2048,512,1,1]",   1,  512,14,14, 2048,1,1,1));
    run(bench_conv("PW 2048->512  [1,2048,14,14]x[512,2048,1,1]",  1, 2048,14,14,  512,1,1,1));

    // ── Memory-bandwidth-bound ────────────────────────────────────────────────
    print_section("LayerNorm  (aten::layer_norm)");
    run(bench_layer_norm("LN [201, 768]    ViT-B per-block x24",    201,  768));
    run(bench_layer_norm("LN [3136, 128]   ConvNeXt stage 1",      3136,  128));
    run(bench_layer_norm("LN [784, 256]    ConvNeXt stage 2",       784,  256));
    run(bench_layer_norm("LN [196, 512]    ConvNeXt stage 3",       196,  512));
    run(bench_layer_norm("LN [49, 1024]    ConvNeXt stage 4",        49, 1024));

    print_section("Softmax  (aten::softmax)");
    run(bench_softmax("Softmax [12*201, 201]  ViT-B attn",     12*201, 201));
    run(bench_softmax("Softmax [12*512, 512]  longer seq ref", 12*512, 512));

    print_section("GELU  (aten::gelu)");
    run(bench_gelu("GELU ConvNeXt MLP  [1,1024,28,28]", 1*1024*28*28));
    run(bench_gelu("GELU ViT-B MLP     [201, 3072]",    201*3072));

    print_section("Elementwise Add  (aten::add -- residual)");
    run(bench_add("Add residual ViT-B   [201, 768]",   201*768));
    run(bench_add("Add residual [1,128,56,56]",        128*56*56));
    run(bench_add("Add residual [1,256,28,28]",        256*28*28));
    run(bench_add("Add residual [1,512,14,14]",        512*14*14));
    run(bench_add("Add residual [1,1024,7,7]",         1024*7*7));

    print_section("Embedding Lookup  (aten::embedding)");
    run(bench_embedding("Embedding seq=128  vocab=32k dim=768",   128, 32768,  768));
    run(bench_embedding("Embedding seq=512  vocab=50k dim=1024",  512, 50257, 1024));
    run(bench_embedding("Embedding seq=2048 vocab=50k dim=1024", 2048, 50257, 1024));

    // ── Output ────────────────────────────────────────────────────────────────
    if (g_json) {
        print_json_output(results);
    } else {
        // Table summary
        std::cout << "\n" << hline() << "\nSUMMARY\n" << hline() << "\n";
        double compute_ms = 0, memory_ms = 0, compute_gf = 0, peak_gb = 0;
        for (auto& r : results) {
            if (r.category == "compute") { compute_ms += r.mean_ms; compute_gf += r.gflops; }
            else                         { memory_ms  += r.mean_ms; peak_gb = std::max(peak_gb, r.gbytes_per_s); }
        }
        std::cout << std::fixed;
        std::cout << "  Total operators       : " << results.size() << "\n";
        std::cout << "  Compute-bound total   : " << std::setprecision(2) << compute_ms
                  << " ms  (" << std::setprecision(1) << compute_gf << " GFLOP/s aggregate)\n";
        std::cout << "  Memory-bound total    : " << std::setprecision(2) << memory_ms  << " ms\n";
        std::cout << "  Peak bandwidth seen   : " << std::setprecision(1) << peak_gb    << " GB/s\n";
        std::cout << "  Grand total           : " << std::setprecision(2)
                  << compute_ms + memory_ms << " ms\n\n";
    }

    return 0;
}