// gelu.cpp
//
// GELU elementwise-activation benchmark on the ggml CPU backend.
//
// NOTE: ggml's CPU kernel for GGML_OP_GELU only implements F32 and F16
// (see ggml/src/ggml-cpu/ops.cpp: ggml_compute_forward_gelu). There is no
// BF16 or quantized GELU kernel on CPU as of this writing, so only f32/f16
// are offered here rather than silently falling back to something else.
//
// Shape is given as rows x cols (e.g. a [n_tokens, n_ff] activation
// tensor coming out of an MLP up-projection).
//
// Usage:
//   ./gelu --rows 1 --cols 14336 --dtype f32 --threads 0 --iters 50 --warmup 10
//
//   --rows     number of rows                                       default 1
//   --cols     number of columns (elements per row)                default 14336
//  --dtype    f32 | f16 | bf16 | q8_0 | q4_0 (weight dtype)      default bf16
//   --threads  0 = use all cores                                  default 0
//   --iters    timed iterations                                    default 50
//   --warmup   untimed warmup iterations                            default 10
//   --seed     RNG seed                                             default 42
//   --no-header  omit the CSV header line (for appending sweep runs)
//
// Output is CSV, one row per timed iteration: op,rows,cols,dtype,threads,iter,ms,gbps

#include "common.h"

int main(int argc, char ** argv) {
    bench::Args args(argc, argv);

    const int64_t rows    = args.get_int("rows", 1);
    const int64_t cols    = args.get_int("cols", 14336);
    const std::string dt  = args.get_str("dtype", "f32");
    const int n_threads   = (int) args.get_int("threads", 0);
    const int n_iters     = (int) args.get_int("iters", 50);
    const int n_warmup    = (int) args.get_int("warmup", 10);
    const unsigned seed   = (unsigned) args.get_int("seed", 42);

    if (dt != "f32" && dt != "f16") {
        fprintf(stderr, "error: gelu only supports dtype f32 or f16 on the CPU backend "
                         "(got '%s'); see the comment at the top of gelu.cpp\n", dt.c_str());
        return 1;
    }
    const enum ggml_type type = bench::parse_type(dt);

    bench::Env env;
    env.init(/*n_tensors=*/2, n_threads);

    struct ggml_tensor * x = ggml_new_tensor_2d(env.ctx, type, cols, rows);
    ggml_set_name(x, "x_input");

    struct ggml_tensor * y = ggml_gelu(env.ctx, x);
    ggml_set_name(y, "y_result");

    env.alloc();

    auto x_data = bench::random_floats((size_t) rows * cols, seed, -5.0f, 5.0f);
    bench::upload_matrix(x, x_data.data(), rows, cols);

    struct ggml_cgraph * graph = ggml_new_graph(env.ctx);
    ggml_build_forward_expand(graph, y);

    auto ms = bench::time_graph(env.backend, graph, n_warmup, n_iters);

    // GELU is memory-bound: report achieved bandwidth (read x + write y).
    const double bytes = (double) rows * cols * ggml_type_size(type) * 2.0;
    const int actual_threads = n_threads > 0 ? n_threads : (int) std::thread::hardware_concurrency();
    const bool header = !args.get_bool("no-header", false);

    bench::print_csv("gelu", {{"rows", std::to_string(rows)}, {"cols", std::to_string(cols)}},
                      dt.c_str(), actual_threads, ms, bytes, "gbps", header);

    env.free();
    return 0;
}
