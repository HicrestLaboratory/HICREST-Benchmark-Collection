// softmax.cpp
//
// Softmax benchmark on the ggml CPU backend, using ggml_soft_max — the same
// op used for attention-score normalization in llama.cpp.
//
// NOTE: ggml's CPU kernel for GGML_OP_SOFT_MAX only implements F32 for the
// main input tensor (see ggml/src/ggml-cpu/ops.cpp:
// ggml_compute_forward_soft_max). There is no BF16/quantized softmax kernel
// on CPU, so this benchmark is f32-only.
//
// Shape is given as rows x cols, e.g. [n_heads * n_tokens, n_kv] for an
// attention-score matrix; softmax is computed independently per row.
//
// Usage:
//   ./softmax --rows 32 --cols 4096 --threads 0 --iters 50 --warmup 10
//
//   --rows     number of independent rows (softmax is per-row)   default 32
//   --cols     row length (the softmax reduction dimension)      default 4096
//   --threads  0 = use all cores                                  default 0
//   --dtype    f32 | f16 | bf16 | q8_0 | q4_0 (weight dtype)      default bf16
//   --iters    timed iterations                                    default 50
//   --warmup   untimed warmup iterations                            default 10
//   --seed     RNG seed                                             default 42
//   --no-header  omit the CSV header line (for appending sweep runs)
//
// Output is CSV, one row per timed iteration: op,rows,cols,dtype,threads,iter,ms,gbps

#include "common.h"

int main(int argc, char ** argv) {
    bench::Args args(argc, argv);

    const int64_t rows  = args.get_int("rows", 32);
    const int64_t cols  = args.get_int("cols", 4096);
    const int n_threads = (int) args.get_int("threads", 0);
    const int n_iters   = (int) args.get_int("iters", 50);
    const int n_warmup  = (int) args.get_int("warmup", 10);
    const unsigned seed = (unsigned) args.get_int("seed", 42);
    const std::string dt  = args.get_str("dtype", "bf16");

    const enum ggml_type type = bench::parse_type(dt);

    bench::Env env;
    env.init(/*n_tensors=*/2, n_threads);

    struct ggml_tensor * x = ggml_new_tensor_2d(env.ctx, type, cols, rows);
    ggml_set_name(x, "x_input");

    struct ggml_tensor * y = ggml_soft_max(env.ctx, x);
    ggml_set_name(y, "y_result");

    env.alloc();

    auto x_data = bench::random_floats((size_t) rows * cols, seed, -10.0f, 10.0f);
    bench::upload_matrix(x, x_data.data(), rows, cols);

    struct ggml_cgraph * graph = ggml_new_graph(env.ctx);
    ggml_build_forward_expand(graph, y);

    auto ms = bench::time_graph(env.backend, graph, n_warmup, n_iters);

    // Memory-bound op: report achieved bandwidth (read x + write y).
    const double bytes = (double) rows * cols * ggml_type_size(type) * 2.0;
    const int actual_threads = n_threads > 0 ? n_threads : (int) std::thread::hardware_concurrency();
    const bool header = !args.get_bool("no-header", false);

    bench::print_csv("softmax", {{"rows", std::to_string(rows)}, {"cols", std::to_string(cols)}},
                      "f32", actual_threads, ms, bytes, "gbps", header);

    env.free();
    return 0;
}
