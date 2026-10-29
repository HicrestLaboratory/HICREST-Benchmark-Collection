# Benchmark Shapes: GEMM, GEMV, Softmax, and GELU

### Notation
- `(M × K) × (K × N) = (M × N)`
- **GEMM / GEMV**: Matrix operations using shapes `M, N, K`.
- **Softmax / GELU**: Applied to the resulting `M × N` dimensions.
- **M values** (tokens per forward pass):
  - **GEMV**: `M = 1` (single-token decoding)
  - **GEMM, Softmax, GELU**: `M ∈ {64, 512, 2048}`

---

### 🟨 Square (`N = K`)

| M, N, K | Model | Paper / Source |
|---|---|---|
| M, 3072, 3072 | Phi-4-mini | [Phi-4-Mini Technical Report](https://arxiv.org/abs/2503.01743) |
| M, 5120, 5120 | Llama 4 Scout | [Model source](https://www.llama.com/models/llama-4/) |
| M, 2880, 2880 | GPT-OSS-120B | [GPT-OSS Model Card](https://arxiv.org/abs/2508.10925) |
 M, 4096, 4096 | Llama 3.1 8B | [Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783), [config](https://huggingface.co/unsloth/Llama-3.1-8B-Instruct/blob/main/config.json) |
 | M, 8192, 8192 | Llama 3.1 70B | [Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783), [config](https://huggingface.co/unsloth/Meta-Llama-3.1-70B-Instruct/blob/main/config.json) |

### 🟩 Short and wide (`N > K`)

| M, N, K | Model | Paper / Source |
|---|---|---|
| M, 6144, 5120 | Qwen3.8-27B | [Qwen3.8 model source](https://huggingface.co/Qwen/Qwen3.8-27B) |
| M, 17408, 5120 | Qwen3.8-27B | [Qwen3.8 model source](https://huggingface.co/Qwen/Qwen3.8-27B) |
| M, 21504, 5376 | Gemma 4 31B | [Gemma 4 Technical Report](https://arxiv.org/abs/2607.02770) |
| M, 8192, 5120 | Llama 4 Scout | [Model source](https://www.llama.com/models/llama-4/) |
| M, 262144, 5376 | Gemma 4 31B | [Gemma 4 Technical Report](https://arxiv.org/abs/2607.02770) |

### 🟦 Tall and narrow (`K > N`)

| M, N, K | Model | Paper / Source |
|---|---|---|
| M, 1024, 5120 | Qwen3.8-27B | [Qwen3.8 model source](https://huggingface.co/Qwen/Qwen3.8-27B) |
| M, 4096, 5376 | Gemma 4 31B | [Gemma 4 Technical Report](https://arxiv.org/abs/2607.02770) |
| M, 512, 2880 | GPT-OSS-120B | [GPT-OSS Model Card](https://arxiv.org/abs/2508.10925) |
| M, 2304, 5120 | DeepSeek V4.1-Flash-1017 | [DeepSeek-V4.1-Flash Technical Report](https://arxiv.org/abs/2609.19969) |
| M, 5376, 21504 | Gemma 4 31B | [Gemma 4 Technical Report](https://arxiv.org/abs/2607.02770) |