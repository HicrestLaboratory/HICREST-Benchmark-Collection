// common.h
//
// Shared helpers for the ggml micro-benchmarks (gemm, gemv, gelu, softmax).
// Everything here is header-only on purpose so each benchmark stays a single
// self-contained .cpp file that links straight against the `ggml` target
// built from the vendored llama.cpp/ggml sources.

#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace bench {

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

struct Timer {
    std::chrono::high_resolution_clock::time_point t0;
    void start() { t0 = std::chrono::high_resolution_clock::now(); }
    double stop_ms() const {
        return std::chrono::duration<double, std::milli>(
                   std::chrono::high_resolution_clock::now() - t0)
            .count();
    }
};

// ---------------------------------------------------------------------------
// dtype handling
//
// NOTE on CPU-backend support in current ggml (checked against the source in
// llama.cpp/ggml/src/ggml-cpu):
//   - mul_mat (GEMM/GEMV) : src0 (the "weight") may be F32, F16, BF16, or any
//                           quantized type (Q8_0, Q4_0, ...). src1 (the
//                           "activation") is F32 here for a realistic
//                           weight-only quantization setup.
//   - gelu                : CPU kernel only implements F32 and F16. There is
//                           no BF16/quantized GELU kernel, so this benchmark
//                           does not offer those options for gelu.
//   - soft_max             : CPU kernel only implements F32.
// ---------------------------------------------------------------------------

inline enum ggml_type parse_type(const std::string & s) {
    if (s == "f32")  return GGML_TYPE_F32;
    if (s == "f16")  return GGML_TYPE_F16;
    if (s == "bf16") return GGML_TYPE_BF16;
    if (s == "q8_0") return GGML_TYPE_Q8_0;
    if (s == "q4_0") return GGML_TYPE_Q4_0;
    fprintf(stderr, "error: unknown dtype '%s' (supported: f32, f16, bf16, q8_0, q4_0)\n", s.c_str());
    exit(1);
}

inline const char * type_name(enum ggml_type t) { return ggml_type_name(t); }

// ---------------------------------------------------------------------------
// Random data
// ---------------------------------------------------------------------------

inline std::vector<float> random_floats(size_t n, unsigned seed, float lo = -1.0f, float hi = 1.0f) {
    std::vector<float> v(n);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (auto & x : v) x = dist(rng);
    return v;
}

// Upload `n_rows` x `n_per_row` fp32 source data into tensor `t`, converting
// to t's storage type (F32 / F16 / BF16 / any block-quantized type).
inline void upload_matrix(struct ggml_tensor * t, const float * src, int64_t n_rows, int64_t n_per_row) {
    const int64_t n = n_rows * n_per_row;
    switch (t->type) {
        case GGML_TYPE_F32: {
            ggml_backend_tensor_set(t, src, 0, (size_t) n * sizeof(float));
            break;
        }
        case GGML_TYPE_F16: {
            std::vector<ggml_fp16_t> tmp(n);
            ggml_fp32_to_fp16_row(src, tmp.data(), n);
            ggml_backend_tensor_set(t, tmp.data(), 0, tmp.size() * sizeof(ggml_fp16_t));
            break;
        }
        case GGML_TYPE_BF16: {
            std::vector<ggml_bf16_t> tmp(n);
            ggml_fp32_to_bf16_row(src, tmp.data(), n);
            ggml_backend_tensor_set(t, tmp.data(), 0, tmp.size() * sizeof(ggml_bf16_t));
            break;
        }
        default: {
            if (!ggml_is_quantized(t->type)) {
                fprintf(stderr, "error: upload_matrix: unsupported tensor type %s\n", ggml_type_name(t->type));
                exit(1);
            }
            std::vector<uint8_t> tmp((size_t) ggml_row_size(t->type, n_per_row) * n_rows);
            ggml_quantize_chunk(t->type, src, tmp.data(), 0, n_rows, n_per_row, nullptr);
            ggml_backend_tensor_set(t, tmp.data(), 0, tmp.size());
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// Minimal CLI parsing: `--flag value` pairs, order independent.
// ---------------------------------------------------------------------------

class Args {
public:
    Args(int argc, char ** argv) {
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a.rfind("--", 0) == 0) {
                std::string key = a.substr(2);
                // Only consume the next token as this flag's value if it
                // isn't itself another `--flag` (lets boolean flags like
                // `--no-header` be followed by another flag safely).
                bool has_val = (i + 1 < argc) && std::string(argv[i + 1]).rfind("--", 0) != 0;
                map_[key] = has_val ? argv[i + 1] : "1";
                if (has_val) ++i;
            }
        }
    }

    bool has(const std::string & key) const { return map_.count(key) != 0; }

    std::string get_str(const std::string & key, const std::string & def) const {
        auto it = map_.find(key);
        return it == map_.end() ? def : it->second;
    }

    int64_t get_int(const std::string & key, int64_t def) const {
        auto it = map_.find(key);
        if (it == map_.end() || it->second.empty()) return def;
        return std::atoll(it->second.c_str());
    }

    // For boolean flags, e.g. `--no-header` (present => true) vs an explicit
    // `--no-header 0` (false).
    bool get_bool(const std::string & key, bool def) const {
        auto it = map_.find(key);
        if (it == map_.end()) return def;
        return it->second != "0" && it->second != "false";
    }

private:
    std::map<std::string, std::string> map_;
};

// ---------------------------------------------------------------------------
// Backend / context bootstrap shared by every benchmark
// ---------------------------------------------------------------------------

struct Env {
    struct ggml_context * ctx     = nullptr;
    ggml_backend_t        backend = nullptr;
    ggml_backend_buffer_t buffer  = nullptr; // set after tensors are declared + alloc_ctx_tensors()

    // `n_tensors` is the max number of leaf/op tensors you will create in ctx;
    // used only to size the (metadata-only) context memory pool.
    void init(int n_tensors, int n_threads) {
        struct ggml_init_params params = {
            /*.mem_size   =*/ (size_t) n_tensors * ggml_tensor_overhead() + ggml_graph_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true, // tensor data lives in the backend buffer, not in ctx
        };
        ctx = ggml_init(params);
        if (!ctx) {
            fprintf(stderr, "error: ggml_init failed\n");
            exit(1);
        }

        backend = ggml_backend_cpu_init();
        if (!backend) {
            fprintf(stderr, "error: ggml_backend_cpu_init failed\n");
            exit(1);
        }
        if (n_threads <= 0) {
            n_threads = (int) std::max(1u, std::thread::hardware_concurrency());
        }
        ggml_backend_cpu_set_n_threads(backend, n_threads);
    }

    // Call once all tensors have been declared in ctx (ggml_new_tensor_*),
    // before uploading data into them.
    void alloc() {
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buffer) {
            fprintf(stderr, "error: ggml_backend_alloc_ctx_tensors failed (out of memory?)\n");
            exit(1);
        }
    }

    void free() {
        if (buffer)  ggml_backend_buffer_free(buffer);
        if (backend) ggml_backend_free(backend);
        if (ctx)     ggml_free(ctx);
    }
};

// Runs `graph` `n_iters` times after `n_warmup` untimed warmup runs, returns
// per-iteration latencies in milliseconds.
inline std::vector<double> time_graph(ggml_backend_t backend, struct ggml_cgraph * graph,
                                       int n_warmup, int n_iters) {
    for (int i = 0; i < n_warmup; ++i) {
        ggml_backend_graph_compute(backend, graph);
    }
    std::vector<double> ms(n_iters);
    Timer t;
    for (int i = 0; i < n_iters; ++i) {
        t.start();
        ggml_backend_graph_compute(backend, graph);
        ms[i] = t.stop_ms();
    }
    return ms;
}

struct Stats {
    double min_ms, max_ms, mean_ms, median_ms;
};

inline Stats compute_stats(std::vector<double> ms) {
    std::sort(ms.begin(), ms.end());
    Stats s;
    s.min_ms    = ms.front();
    s.max_ms    = ms.back();
    s.median_ms = ms[ms.size() / 2];
    double sum  = 0;
    for (double v : ms) sum += v;
    s.mean_ms = sum / ms.size();
    return s;
}

inline void print_result(const char * op, const char * shape_desc, const char * dtype,
                          int n_threads, const Stats & s, double flops_or_bytes_per_iter,
                          const char * unit /* "GFLOPS" or "GB/s" */) {
    double best_ms  = s.min_ms;
    double throughput = flops_or_bytes_per_iter / (best_ms * 1.0e6); // per-ms -> per-s, /1e9 -> giga
    printf("op=%-8s shape=%-24s dtype=%-6s threads=%-3d "
           "min_ms=%-9.4f mean_ms=%-9.4f median_ms=%-9.4f max_ms=%-9.4f %s=%.3f\n",
           op, shape_desc, dtype, n_threads,
           s.min_ms, s.mean_ms, s.median_ms, s.max_ms, unit, throughput);
}

// ---------------------------------------------------------------------------
// CSV output: one row per timed iteration (not just a summary), so every
// sample is kept for downstream analysis (percentiles, variance, plots, ...).
//
// `shape_cols` are the op-specific dimensions (e.g. {{"M","32"},{"K","4096"}}),
// printed as their own CSV columns. `metric_name` is "gflops" or "gbps"; its
// per-iteration value is derived from `work_per_iter` (FLOPs or bytes moved
// per call) divided by that iteration's latency.
// ---------------------------------------------------------------------------

inline void print_csv(const char * op,
                       const std::vector<std::pair<std::string, std::string>> & shape_cols,
                       const char * dtype, int n_threads,
                       const std::vector<double> & ms_per_iter,
                       double work_per_iter, const char * metric_name,
                       bool print_header = true) {
    if (print_header) {
        printf("op,");
        for (auto & c : shape_cols) printf("%s,", c.first.c_str());
        printf("dtype,threads,iter,ms,%s\n", metric_name);
    }
    for (size_t i = 0; i < ms_per_iter.size(); ++i) {
        printf("%s,", op);
        for (auto & c : shape_cols) printf("%s,", c.second.c_str());
        const double metric = work_per_iter / (ms_per_iter[i] * 1.0e6); // work/ms -> giga-work/s
        printf("%s,%d,%zu,%.6f,%.6f\n", dtype, n_threads, i, ms_per_iter[i], metric);
    }
}

} // namespace bench
