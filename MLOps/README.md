# ggml op + model benchmarks

CPU benchmarks against `ggml`/`llama.cpp`, in two independent pieces:

- **`src/`** — op-level microbenchmarks (GEMM, GEMV, GELU, softmax), one
  `.cpp` per op, linked directly against `ggml`.
- **`model-bench/`** — full-model (prompt-processing / token-generation)
  benchmarking, wrapping `llama-bench`.

## Layout

```
ggml-bench/
├── CMakeLists.txt          # builds src/ only (needs ggml)
├── build.sh                 # configure + build src/, with backend selection
├── src/                      # gemm.cpp gemv.cpp gelu.cpp softmax.cpp common.h
├── model-bench/
│   └── run_model_bench.sh     # wraps llama-bench -> per-iteration CSV (needs jq)
└── llama.cpp/                  # <- your clone/submodule goes here
```

`llama.cpp` sits next to `src/`/`model-bench/`, not inside either — symlink
it in if it lives elsewhere: `ln -s /path/to/llama.cpp ggml-bench/llama.cpp`.

## Build the op microbenchmarks

```bash
./build.sh                        # cpu backend — ggml's own kernels (default)
./build.sh --backend blas         # system BLAS, auto-discovered
./build.sh --backend blis \
    --blas-lib /path/libblis.so --blas-include /path/include   # specific local build
```

Produces `build/{gemm,gemv,gelu,softmax}`, without touching `llama.cpp`'s
CLI tools/server/tests. Extra cmake flags (e.g. `-DGGML_NATIVE=OFF` for a
portable build) go after a trailing `--`. Note: `--backend` only changes
what gets *built* — the benchmarks call `ggml_backend_cpu_init()`, so BLAS/
BLIS isn't exercised at runtime yet without a further code change.

## Run

```bash
./build/gemm    --m 32 --k 4096 --n 4096 --dtype bf16 --iters 20   # C[M,N]=B[M,K]xA[K,N]
./build/gemv    --k 4096 --n 4096 --dtype q8_0 --iters 50          # batch-1 GEMM
./build/gelu    --rows 1 --cols 14336 --dtype f32 --iters 50
./build/softmax --rows 32 --cols 4096 --iters 50
```

`--threads 0` (default) = all cores. Output is CSV, one row per **timed
iteration** (warmup excluded), not just a mean:

```
op,M,K,N,dtype,threads,iter,ms,gflops
gemm,32,4096,4096,bf16,8,0,1.2044,27.85
```

`gemv` → `K,N`; `gelu`/`softmax` → `rows,cols` + `gbps` instead of `gflops`.
Build a sweep with `--no-header` after the first call:

```bash
./gemm --dtype f32  --iters 20            > sweep.csv
./gemm --dtype bf16 --iters 20 --no-header >> sweep.csv
```

### Dtype support

Checked against `ggml/src/ggml-cpu/*.cpp` — not a limitation of the
benchmark, just what ggml's CPU backend implements today:

| op | supported |
|---|---|
| gemm / gemv | weight: `f32`, `f16`, `bf16`, `q8_0`, `q4_0` (activation always `f32`) |
| gelu | `f32`, `f16` only |
| softmax | `f32` only |

"int8" = `q8_0`, ggml's block-quantized format (32-element blocks + fp16
scale) — what `llama.cpp` itself uses; there's no unquantized raw-int8
kernel. `K` must be a multiple of the dtype's block size (32 for
`q8_0`/`q4_0`, 1 otherwise). An unsupported `--dtype` errors clearly instead
of silently upcasting. Timing wraps `ggml_backend_graph_compute` only.

## Full-model benchmarking

No custom harness here — `llama-bench` already handles GGUF loading, KV
cache, batching, and pp/tg timing correctly for every architecture
`llama.cpp` supports, so `model-bench/run_model_bench.sh` just wraps it
instead of re-implementing that (and risking getting the methodology wrong).

Build `llama-bench` (separate from this project's own build):

```bash
cd llama.cpp && cmake -B build && cmake --build build --target llama-bench -j
```

`llama-bench`'s own `-o csv`/`-o md` only print aggregated avg/stddev; only
`-o json`/`-o jsonl` carry every repetition. `run_model_bench.sh` runs
`llama-bench -o jsonl` directly (no shell command construction) and
explodes it into one CSV row per iteration via `jq` (required):

```bash
./model-bench/run_model_bench.sh \
    --llama-bench llama.cpp/build/bin/llama-bench \
    --out model_bench.csv -- \
    -m model.gguf -p 512 -n 128 -t 4,8 -r 10
```

Everything after `--` passes straight to `llama-bench` (same flags as
`--help`); don't pass `-o` yourself. Output columns:

```
model_filename,n_prompt,n_gen,n_threads,n_batch,n_ubatch,type_k,type_v,flash_attn,n_gpu_layers,test_time,iter,ns,ms,tokens_per_sec
```