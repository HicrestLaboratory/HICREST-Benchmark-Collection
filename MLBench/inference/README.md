# CPU AI Inference
Inference of small models using llama.cpp (llama-bench).

## 1. Compile llama.cpp

From the project root (one level above this folder):

```bash
./compile_llama.sh
```

This builds `build/bin/llama-bench`. `run.sh` expects that path by default;
pass `--llama-bench /path/to/llama-bench` if yours lives elsewhere.

## 2. Download the models

```bash
./download-models.sh
```

Creates `models/` next to this script and fetches all four `.gguf` files
(skips any that are already present). Total download is roughly 9 GB.

## 3. Run the benchmark

Everything after `--` is passed straight to `llama-bench`:
- `-m` model file
- `-p` prompt-processing (prefill) tokens tested — set to `0` to skip prefill (compute bound)
- `-n` text-generation (decode) tokens tested — set to `0` to skip decode (memory bound)
- `-t` thread count(s) to test
- `-r` repetitions per setting

### Prefill only (`-n 0`)

```bash
./run.sh -- -m models/granite-4.0-h-micro-Q4_K_M.gguf -p 512 -n 0 -t 8 -r 10
./run.sh -- -m models/mamba-2.8b-hf-q4_k_m.gguf -p 512 -n 0 -t 8 -r 10
./run.sh -- -m models/google_gemma-4-E2B-it-Q4_K_M.gguf -p 512 -n 0 -t 8 -r 10
./run.sh -- -m models/granite-3.1-3b-a800m-instruct-Q4_K_M.gguf -p 512 -n 0 -t 8 -r 10
```

### Decode only (`-p 0`)

```bash
./run.sh -- -m models/granite-4.0-h-micro-Q4_K_M.gguf -p 0 -n 128 -t 8 -r 10
./run.sh -- -m models/mamba-2.8b-hf-q4_k_m.gguf -p 0 -n 128 -t 8 -r 10
./run.sh -- -m models/google_gemma-4-E2B-it-Q4_K_M.gguf -p 0 -n 128 -t 8 -r 10
./run.sh -- -m models/granite-3.1-3b-a800m-instruct-Q4_K_M.gguf -p 0 -n 128 -t 8 -r 10
```

### Prefill + decode together (single combined pass)

`-pg <pp,tg>` runs the prompt and the generation back-to-back as one
continuous pass and reports it as a single `pp+tg` row (same idea as the
`tg` row, but covering both phases together) — this is different from just
setting `-p` and `-n` in the same call, which runs them as two separate,
independent tests.

```bash
./run.sh -- -m models/granite-4.0-h-micro-Q4_K_M.gguf -p 0 -n 0 -pg 512,128 -t 8 -r 10
./run.sh -- -m models/mamba-2.8b-hf-q4_k_m.gguf -p 0 -n 0 -pg 512,128 -t 8 -r 10
./run.sh -- -m models/google_gemma-4-E2B-it-Q4_K_M.gguf -p 0 -n 0 -pg 512,128 -t 8 -r 10
./run.sh -- -m models/granite-3.1-3b-a800m-instruct-Q4_K_M.gguf -p 0 -n 0 -pg 512,128 -t 8 -r 10
```