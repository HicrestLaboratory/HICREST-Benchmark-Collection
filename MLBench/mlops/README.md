# MLOps Micro-Benchmarks

CPU micro-benchmarks of the ggml kernels used by llama.cpp: `gemm`, `gemv`,
`gelu`, `softmax`. Each prints CSV results to stdout (one row per iteration).

## Build

From the `MLBench/` folder:

```bash
./compile_llama.sh
```

Binaries: `build/mlops/{gemm,gemv,gelu,softmax}`.

## Common parameters (all four)

| Flag        | Values                                   | Default | Meaning |
|-------------|------------------------------------------|---------|---------|
| `--threads` | any positive integer, or `0`             | `0`     | `0` = all cores. |
| `--iters`   | any positive integer                     | `20` (gemm), `50` (others) | Timed iterations. |
| `--warmup`  | any positive integer                     | `5` (gemm), `10` (others) | Untimed warmup runs. |
| `--seed`    | any integer                              | `42`    | RNG seed. |

## gemm — matrix multiplication `(M×K) × (K×N)`

| Flag      | Values                   | Default |
|-----------|--------------------------|---------|
| `--m`     | any positive integer     | `1`     |
| `--k`     | any positive integer     | `4096`  |
| `--n`     | any positive integer     | `4096`  |
| `--dtype` | `f32`, `f16`, `bf16`, `q8_0` | `bf16` |

- With `q8_0`, `k` must be a multiple of 32.

```bash
./build/mlops/gemm --m 1 --k 4096 --n 4096 --dtype bf16 --threads 8 --iters 20
```

## gemv — matrix-vector product (same op, one input vector)

| Flag      | Values                       | Default |
|-----------|------------------------------|---------|
| `--k`     | any positive integer         | `4096`  |
| `--n`     | any positive integer         | `4096`  |
| `--dtype` | `f32`, `f16`, `bf16`, `q8_0` | `bf16`  |

- With `q8_0`, `k` must be a multiple of 32.

```bash
./build/mlops/gemv --k 4096 --n 4096 --dtype q8_0 --threads 8 --iters 50
```

## gelu — elementwise activation

| Flag      | Values               | Default |
|-----------|----------------------|---------|
| `--rows`  | any positive integer | `1`     |
| `--cols`  | any positive integer | `14336` |
| `--dtype` | `f32`, `f16` only    | `f32`   |

Any other dtype **crashes**.

```bash
./build/mlops/gelu --rows 1 --cols 14336 --dtype f32 --threads 8 --iters 50
```

## softmax

| Flag      | Values            | Default |
|-----------|-------------------|---------|
| `--rows`  | any positive integer | `32`  |
| `--cols`  | any positive integer | `4096` |
| `--dtype` | `f32` only        | `f32`   |

Only `f32` works; any other value **crashes**. Leave `--dtype` unset.

```bash
./build/mlops/softmax --rows 32 --cols 4096 --threads 8 --iters 50
```
