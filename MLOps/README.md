# ggml op benchmarks (GEMM / GEMV / GELU / Softmax)

Four standalone CPU benchmarks that link directly against `ggml`, the tensor
library that ships inside `llama.cpp`. Each benchmark is one `.cpp` file, one
op, and takes its problem size from the command line.

## Layout

Your own `llama.cpp` clone sits **next to** `src/`, not inside it — the build
treats it as an external dependency and only compiles its `ggml/` subfolder
(not the `llama` library, server, or examples):

```
ggml-bench/
├── CMakeLists.txt
├── README.md
├── src/
│   ├── common.h      # shared timing / CLI / dtype-upload helpers
│   ├── gemm.cpp
│   ├── gemv.cpp
│   ├── gelu.cpp
│   └── softmax.cpp
└── llama.cpp/        # <- put/symlink your existing clone here
```

If your clone lives somewhere else, either move it, or symlink it:

```bash
ln -s /path/to/your/llama.cpp /path/to/ggml-bench/llama.cpp
```

(Or edit the single `add_subdirectory(...)` path near the top of
`CMakeLists.txt` if you'd rather point at it directly.)

## Build

```bash
cd ggml-bench
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..     # add -DGGML_NATIVE=OFF if building
                                         # for a different CPU than you run on
cmake --build . -j
```

This produces four executables: `gemm`, `gemv`, `gelu`, `softmax`.
`add_subdirectory` only pulls in `ggml`'s own build graph, so this does not
build `llama.cpp`'s CLI tools, server, or test suite — configure is fast and
the only slow part is compiling ggml's CPU kernels once.

## Run

Every flag is `--name value`, all have defaults, order doesn't matter.

```bash
# GEMM: C[M,N] = B[M,K] x A[K,N]   (A = "weight", quantizable; B = f32 activations)
./gemm --m 32 --k 4096 --n 4096 --dtype bf16 --threads 0 --iters 20 --warmup 5

# GEMV: y[N] = W[K,N]^T x[K]       (batch-1 GEMM, the decode-time op)
./gemv --k 4096 --n 4096 --dtype q8_0 --threads 0 --iters 50 --warmup 10

# GELU: elementwise, rows x cols
./gelu --rows 1 --cols 14336 --dtype f32 --threads 0 --iters 50 --warmup 10

# Softmax: per-row, rows x cols
./softmax --rows 32 --cols 4096 --threads 0 --iters 50 --warmup 10
```

`--threads 0` (the default) uses all detected cores. Output is CSV, one row
per **timed iteration** (warmup rows are not included) — every sample is
kept, not just a mean:

```
op,M,K,N,dtype,threads,iter,ms,gflops
gemm,32,4096,4096,bf16,8,0,1.2044,27.85
gemm,32,4096,4096,bf16,8,1,1.1998,27.96
...
```

`gemv` uses `K,N`; `gelu`/`softmax` use `rows,cols`, with a `gbps` column
instead of `gflops`. Redirect a single run to a file, or build up a sweep
across multiple dtypes/shapes with `--no-header` on every call after the
first:

```bash
./gemm --k 4096 --n 4096 --dtype f32  --iters 20            > sweep.csv
./gemm --k 4096 --n 4096 --dtype bf16 --iters 20 --no-header >> sweep.csv
./gemm --k 4096 --n 4096 --dtype q8_0 --iters 20 --no-header >> sweep.csv
```

## Datatype support — why it isn't uniform across the four ops

This isn't a limitation of the benchmark; it's what ggml's CPU backend
actually implements today (checked against `ggml/src/ggml-cpu/*.cpp` in the
cloned repo):

| op       | supported dtypes on CPU                              |
|----------|-------------------------------------------------------|
| gemm     | weight: `f32`, `f16`, `bf16`, `q8_0`, `q4_0`; activation is always `f32` |
| gemv     | same as gemm                                          |
| gelu     | `f32`, `f16` only — no BF16/quantized GELU CPU kernel  |
| softmax  | `f32` only — no BF16/quantized softmax CPU kernel      |

For GEMM/GEMV, "int8" is represented as `q8_0` — ggml's block-quantized int8
format (32-element blocks, each with an fp16 scale), which is what
`llama.cpp` actually uses for int8 weight quantization; there's no
scale-less raw-int8 matmul kernel to benchmark instead. `q4_0` is included
too since it's the same code path and a natural comparison point.
`gelu.cpp`/`softmax.cpp` will refuse an unsupported `--dtype` with a clear
error rather than silently upcasting.

## Notes on the numbers

- `M` in gemm is the activation batch size; `gemm --m 1` is numerically the
  same op as `gemv`, but `gemv.cpp` is kept separate because ggml's CPU
  kernels take a distinct (memory-bandwidth-bound) path for batch=1 vs. a
  compute-bound batched GEMM, and that's usually the interesting comparison.
- `K` must be a multiple of the dtype's block size (32 for `q8_0`/`q4_0`, 1
  for `f32`/`f16`/`bf16`); the benchmark checks this and exits with a clear
  message rather than crashing.
- Timing wraps `ggml_backend_graph_compute` only (warmup iterations excluded)
  — it does not include the one-time tensor allocation/quantization done
  before the graph is built.