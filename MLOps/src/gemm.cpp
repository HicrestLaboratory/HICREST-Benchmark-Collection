// gemm.cpp
//
// GEMM benchmark on the ggml CPU backend, using the same op (ggml_mul_mat)
// that llama.cpp uses for every linear layer.
//
//   A = "weight"     [K, N]   (dtype selectable: f32, f16, bf16, q8_0, q4_0)
//   B = "activation"  [K, M]   (f32)
//   C = A^T @ per-row(B)  ->  [N, M], i.e. an (M x K) x (K x N) = (M x N) GEMM
//
// Usage:
//   ./gemm --m 1 --k 4096 --n 4096 --dtype bf16 --threads 0 --iters 20 --warmup 5
//
//   --m        rows of the activation matrix (batch size)       default 1
//   --k        shared/reduction dimension                        default 4096
//   --n        output dimension (rows of the weight matrix)       default 4096
//   --dtype    f32 | f16 | bf16 | q8_0 | q4_0 (weight dtype)      default bf16
//   --threads  0 = use all cores                                  default 0
//   --iters    timed iterations                                    default 20
//   --warmup   untimed warmup iterations                            default 5
//   --seed     RNG seed                                             default 42
//   --no-header  omit the CSV header line (for appending sweep runs)
//
// Output is CSV, one row per timed iteration: op,M,K,N,dtype,threads,iter,ms,gflops
// Redirect to a file, or append multiple runs with `--no-header` after the first:
//   ./gemm --k 4096 --n 4096 --dtype bf16  > out.csv
//   ./gemm --k 4096 --n 4096 --dtype q8_0 --no-header >> out.csv

#include "common.h"

int main(int argc, char ** argv) {
    bench::Args args(argc, argv);

    const int64_t M       = args.get_int("m", 1);
    const int64_t K       = args.get_int("k", 4096);
    const int64_t N       = args.get_int("n", 4096);
    const std::string dt  = args.get_str("dtype", "bf16");
    const int n_threads   = (int) args.get_int("threads", 0);
    const int n_iters     = (int) args.get_int("iters", 20);
    const int n_warmup    = (int) args.get_int("warmup", 5);
    const unsigned seed   = (unsigned) args.get_int("seed", 42);

    const enum ggml_type wtype = bench::parse_type(dt); // weight dtype
    const enum ggml_type atype = bench::parse_type(dt); //GGML_TYPE_F32;          // activation dtype

    if (K % ggml_blck_size(wtype) != 0) {
        fprintf(stderr, "error: k=%lld must be a multiple of the block size (%lld) for dtype %s\n",
                (long long) K, (long long) ggml_blck_size(wtype), dt.c_str());
        return 1;
    }

    bench::Env env;
    env.init(/*n_tensors=*/4, n_threads);

    struct ggml_tensor * a = ggml_new_tensor_2d(env.ctx, wtype, K, N); // weight   [K, N]
    struct ggml_tensor * b = ggml_new_tensor_2d(env.ctx, atype, K, M); // activ.   [K, M]
    ggml_set_name(a, "A_weight");
    ggml_set_name(b, "B_activation");

    struct ggml_tensor * c = ggml_mul_mat(env.ctx, a, b); // [N, M]
    ggml_set_name(c, "C_result");

    env.alloc();

    // Fill with random data.
    auto a_data = bench::random_floats((size_t) K * N, seed);
    auto b_data = bench::random_floats((size_t) K * M, seed + 1);
    bench::upload_matrix(a, a_data.data(), N, K);
    bench::upload_matrix(b, b_data.data(), M, K);

    struct ggml_cgraph * graph = ggml_new_graph(env.ctx);
    ggml_build_forward_expand(graph, c);

    auto ms = bench::time_graph(env.backend, graph, n_warmup, n_iters);

    // GEMM FLOPs: 2 * M * N * K (multiply + add per MAC)
    const double flops = 2.0 * (double) M * (double) N * (double) K;
    const int actual_threads = n_threads > 0 ? n_threads : (int) std::thread::hardware_concurrency();
    const bool header = !args.get_bool("no-header", false);

    bench::print_csv("gemm",
                      {{"M", std::to_string(M)}, {"K", std::to_string(K)}, {"N", std::to_string(N)}},
                      dt.c_str(), actual_threads, ms, flops, "gflops", header);

    env.free();
    return 0;
}
