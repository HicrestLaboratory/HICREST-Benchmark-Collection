// gemv.cpp
//
// GEMV benchmark on the ggml CPU backend: a weight matrix times a single
// activation vector, i.e. ggml_mul_mat with the activation batch size fixed
// to 1. This is the exact op used for single-token (batch=1) decoding in
// llama.cpp, and it exercises a different (memory-bandwidth-bound) code path
// in ggml's CPU kernels than the compute-bound batched GEMM case.
//
//   W = "weight"     [K, N]   (dtype selectable: f32, f16, bf16, q8_0, q4_0)
//   x = "activation"  [K]      (f32)
//   y = W^T @ x        [N]
//
// Usage:
//   ./gemv --k 4096 --n 4096 --dtype q8_0 --threads 0 --iters 50 --warmup 10
//
//   --k        input/reduction dimension                          default 4096
//   --n        output dimension                                    default 4096
//   --dtype    f32 | f16 | bf16 | q8_0 | q4_0 (weight dtype)      default bf16
//   --threads  0 = use all cores                                  default 0
//   --iters    timed iterations                                    default 50
//   --warmup   untimed warmup iterations                            default 10
//   --seed     RNG seed                                             default 42
//   --no-header  omit the CSV header line (for appending sweep runs)
//
// Output is CSV, one row per timed iteration: op,K,N,dtype,threads,iter,ms,gflops

#include "common.h"

int main(int argc, char ** argv) {
    bench::Args args(argc, argv);

    const int64_t K       = args.get_int("k", 4096);
    const int64_t N       = args.get_int("n", 4096);
    const std::string dt  = args.get_str("dtype", "bf16");
    const int n_threads   = (int) args.get_int("threads", 0);
    const int n_iters     = (int) args.get_int("iters", 50);
    const int n_warmup    = (int) args.get_int("warmup", 10);
    const unsigned seed   = (unsigned) args.get_int("seed", 42);

    const enum ggml_type wtype = bench::parse_type(dt);
    const enum ggml_type atype = GGML_TYPE_F32;

    if (K % ggml_blck_size(wtype) != 0) {
        fprintf(stderr, "error: k=%lld must be a multiple of the block size (%lld) for dtype %s\n",
                (long long) K, (long long) ggml_blck_size(wtype), dt.c_str());
        return 1;
    }

    bench::Env env;
    env.init(/*n_tensors=*/4, n_threads);

    struct ggml_tensor * w = ggml_new_tensor_2d(env.ctx, wtype, K, N); // weight [K, N]
    struct ggml_tensor * x = ggml_new_tensor_1d(env.ctx, atype, K);    // vector [K]
    ggml_set_name(w, "W_weight");
    ggml_set_name(x, "x_vector");

    struct ggml_tensor * y = ggml_mul_mat(env.ctx, w, x); // [N]
    ggml_set_name(y, "y_result");

    env.alloc();

    auto w_data = bench::random_floats((size_t) K * N, seed);
    auto x_data = bench::random_floats((size_t) K, seed + 1);
    bench::upload_matrix(w, w_data.data(), N, K);
    bench::upload_matrix(x, x_data.data(), 1, K);

    struct ggml_cgraph * graph = ggml_new_graph(env.ctx);
    ggml_build_forward_expand(graph, y);

    auto ms = bench::time_graph(env.backend, graph, n_warmup, n_iters);

    // GEMV FLOPs: 2 * N * K
    const double flops = 2.0 * (double) N * (double) K;
    const int actual_threads = n_threads > 0 ? n_threads : (int) std::thread::hardware_concurrency();
    const bool header = !args.get_bool("no-header", false);

    bench::print_csv("gemv", {{"K", std::to_string(K)}, {"N", std::to_string(N)}},
                      dt.c_str(), actual_threads, ms, flops, "gflops", header);

    env.free();
    return 0;
}
