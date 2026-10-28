# GEMM, GEMV, Softmax and GELU benchmark shapes

Notation: `A[M,K] @ B[K,N] -> C[M,N]`. GEMM uses `M ∈ {128, 512, 2048}` (TO CHECK). GEMV uses `M = 1` for single-token decoding. Softmax and GELU use `C[M,N]` from the corresponding listed shape.

## 1. GEMM

### 🟨 Square (`N = K`)

| M, N, K | Model | Matrix | Paper / source |
|---|---|---|---|
| M, 3072, 3072 | [Phi-4-mini](https://huggingface.co/microsoft/Phi-4-mini-instruct/blob/main/config.json) | Attention Q projection | [Phi-4-Mini Technical Report](https://arxiv.org/abs/2503.01743) |
| M, 3072, 3072 | [Phi-4-mini](https://huggingface.co/microsoft/Phi-4-mini-instruct/blob/main/config.json) | Attention output projection | [Phi-4-Mini Technical Report](https://arxiv.org/abs/2503.01743) |
| M, 5120, 5120 | [Llama 4 Scout](https://www.llama.com/models/llama-4/) | Attention Q projection | [Model source](https://www.llama.com/models/llama-4/) |
| M, 5120, 5120 | [Llama 4 Scout](https://www.llama.com/models/llama-4/) | Attention output projection | [Model source](https://www.llama.com/models/llama-4/) |
| M, 2880, 2880 | [GPT-OSS-120B](https://huggingface.co/openai/gpt-oss-120b/blob/main/config.json) | Expert MLP projection | [GPT-OSS Model Card](https://arxiv.org/abs/2508.10925) |

### 🟩 Tall and thin (`N > K`)

| M, N, K | Model | Matrix | Paper / source |
|---|---|---|---|
| M, 6144, 5120 | [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json) | Attention Q projection | [Qwen3.8 model source](https://huggingface.co/Qwen/Qwen3.8-27B) |
| M, 17408, 5120 | [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json) | MLP up projection | [Qwen3.8 model source](https://huggingface.co/Qwen/Qwen3.8-27B) |
| M, 21504, 5376 | [Gemma 4 31B](https://huggingface.co/google/gemma-4-31B-it/blob/main/config.json) | MLP up projection | [Gemma 4 Technical Report](https://arxiv.org/abs/2607.02770) |
| M, 8192, 5120 | [Llama 4 Scout](https://www.llama.com/models/llama-4/) | Expert MLP up projection | [Model source](https://www.llama.com/models/llama-4/) |
| M, 262144, 5376 | [Gemma 4 31B](https://huggingface.co/google/gemma-4-31B-it/blob/main/config.json) | LM head / vocabulary projection | [Gemma 4 Technical Report](https://arxiv.org/abs/2607.02770) |

### 🟦 Short and fat (`K > N`)

| M, N, K | Model | Matrix | Paper / source |
|---|---|---|---|
| M, 1024, 5120 | [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json) | Attention K/V projection | [Qwen3.8 model source](https://huggingface.co/Qwen/Qwen3.8-27B) |
| M, 4096, 5376 | [Gemma 4 31B](https://huggingface.co/google/gemma-4-31B-it/blob/main/config.json) | Sliding-attention K/V projection | [Gemma 4 Technical Report](https://arxiv.org/abs/2607.02770) |
| M, 512, 2880 | [GPT-OSS-120B](https://huggingface.co/openai/gpt-oss-120b/blob/main/config.json) | Attention K/V projection | [GPT-OSS Model Card](https://arxiv.org/abs/2508.10925) |
| M, 2304, 5120 | [DeepSeek V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/config.json) | MoE expert projection | [DeepSeek-V4.1-Flash Technical Report](https://arxiv.org/abs/2609.19969) |
| M, 5376, 21504 | [Gemma 4 31B](https://huggingface.co/google/gemma-4-31B-it/blob/main/config.json) | MLP down projection | [Gemma 4 Technical Report](https://arxiv.org/abs/2607.02770) |

## 2. GEMV — single-token decoding (`M = 1`)

### 🟨 Square (`N = K`)

| M, N, K | Model | Matrix | Paper / source |
|---|---|---|---|
| 1, 3072, 3072 | [Phi-4-mini](https://huggingface.co/microsoft/Phi-4-mini-instruct/blob/main/config.json) | Attention Q projection | [Phi-4-Mini Technical Report](https://arxiv.org/abs/2503.01743) |
| 1, 3072, 3072 | [Phi-4-mini](https://huggingface.co/microsoft/Phi-4-mini-instruct/blob/main/config.json) | Attention output projection | [Phi-4-Mini Technical Report](https://arxiv.org/abs/2503.01743) |
| 1, 5120, 5120 | [Llama 4 Scout](https://www.llama.com/models/llama-4/) | Attention Q projection | [Model source](https://www.llama.com/models/llama-4/) |
| 1, 5120, 5120 | [Llama 4 Scout](https://www.llama.com/models/llama-4/) | Attention output projection | [Model source](https://www.llama.com/models/llama-4/) |
| 1, 2880, 2880 | [GPT-OSS-120B](https://huggingface.co/openai/gpt-oss-120b/blob/main/config.json) | Expert MLP projection | [GPT-OSS Model Card](https://arxiv.org/abs/2508.10925) |

### 🟩 Tall and thin (`N > K`)

| M, N, K | Model | Matrix | Paper / source |
|---|---|---|---|
| 1, 6144, 5120 | [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json) | Attention Q projection | [Qwen3.8 model source](https://huggingface.co/Qwen/Qwen3.8-27B) |
| 1, 17408, 5120 | [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json) | MLP up projection | [Qwen3.8 model source](https://huggingface.co/Qwen/Qwen3.8-27B) |
| 1, 21504, 5376 | [Gemma 4 31B](https://huggingface.co/google/gemma-4-31B-it/blob/main/config.json) | MLP up projection | [Gemma 4 Technical Report](https://arxiv.org/abs/2607.02770) |
| 1, 8192, 5120 | [Llama 4 Scout](https://www.llama.com/models/llama-4/) | Expert MLP up projection | [Model source](https://www.llama.com/models/llama-4/) |
| 1, 262144, 5376 | [Gemma 4 31B](https://huggingface.co/google/gemma-4-31B-it/blob/main/config.json) | LM head / vocabulary projection | [Gemma 4 Technical Report](https://arxiv.org/abs/2607.02770) |

### 🟦 Short and fat (`K > N`)

| M, N, K | Model | Matrix | Paper / source |
|---|---|---|---|
| 1, 1024, 5120 | [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json) | Attention K/V projection | [Qwen3.8 model source](https://huggingface.co/Qwen/Qwen3.8-27B) |
| 1, 4096, 5376 | [Gemma 4 31B](https://huggingface.co/google/gemma-4-31B-it/blob/main/config.json) | Sliding-attention K/V projection | [Gemma 4 Technical Report](https://arxiv.org/abs/2607.02770) |
| 1, 512, 2880 | [GPT-OSS-120B](https://huggingface.co/openai/gpt-oss-120b/blob/main/config.json) | Attention K/V projection | [GPT-OSS Model Card](https://arxiv.org/abs/2508.10925) |
| 1, 2304, 5120 | [DeepSeek V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/config.json) | MoE expert projection | [DeepSeek-V4.1-Flash Technical Report](https://arxiv.org/abs/2609.19969) |
| 1, 5376, 21504 | [Gemma 4 31B](https://huggingface.co/google/gemma-4-31B-it/blob/main/config.json) | MLP down projection | [Gemma 4 Technical Report](https://arxiv.org/abs/2607.02770) |

## 3. Softmax

### 🟨 Square (`N = K`)

| Source shape (M, N, K) | Softmax matrix | Model | Paper / source |
|---|---|---|---|
| M, 3072, 3072 | `C[M, 3072]` | Phi-4-mini | [Phi-4-Mini Technical Report](https://arxiv.org/abs/2503.01743) |
| M, 3072, 3072 | `C[M, 3072]` | Phi-4-mini | [Phi-4-Mini Technical Report](https://arxiv.org/abs/2503.01743) |
| M, 5120, 5120 | `C[M, 5120]` | Llama 4 Scout | [Model source](https://www.llama.com/models/llama-4/) |
| M, 5120, 5120 | `C[M, 5120]` | Llama 4 Scout | [Model source](https://www.llama.com/models/llama-4/) |
| M, 2880, 2880 | `C[M, 2880]` | GPT-OSS-120B | [GPT-OSS Model Card](https://arxiv.org/abs/2508.10925) |

### 🟩 Tall and thin (`N > K`)

| Source shape (M, N, K) | Softmax matrix | Model | Paper / source |
|---|---|---|---|
| M, 6144, 5120 | `C[M, 6144]` | Qwen3.8-27B | [Qwen3.8 model source](https://huggingface.co/Qwen/Qwen3.8-27B) |
| M, 17408, 5120 | `C[M, 17408]` | Qwen3.8-27B | [Qwen3.8 model source](https://huggingface.co/Qwen/Qwen3.8-27B) |
| M, 21504, 5376 | `C[M, 21504]` | Gemma 4 31B | [Gemma 4 Technical Report](https://arxiv.org/abs/2607.02770) |
| M, 8192, 5120 | `C[M, 8192]` | Llama 4 Scout | [Model source](https://www.llama.com/models/llama-4/) |
| M, 262144, 5376 | `C[M, 262144]` | Gemma 4 31B | [Gemma 4 Technical Report](https://arxiv.org/abs/2607.02770) |

### 🟦 Short and fat (`K > N`)

| Source shape (M, N, K) | Softmax matrix | Model | Paper / source |
|---|---|---|---|
| M, 1024, 5120 | `C[M, 1024]` | Qwen3.8-27B | [Qwen3.8 model source](https://huggingface.co/Qwen/Qwen3.8-27B) |
| M, 4096, 5376 | `C[M, 4096]` | Gemma 4 31B | [Gemma 4 Technical Report](https://arxiv.org/abs/2607.02770) |
| M, 512, 2880 | `C[M, 512]` | GPT-OSS-120B | [GPT-OSS Model Card](https://arxiv.org/abs/2508.10925) |
| M, 2304, 5120 | `C[M, 2304]` | DeepSeek V4.1-Flash-1017 | [DeepSeek-V4.1-Flash Technical Report](https://arxiv.org/abs/2609.19969) |
| M, 5376, 21504 | `C[M, 5376]` | Gemma 4 31B | [Gemma 4 Technical Report](https://arxiv.org/abs/2607.02770) |

## 4. GELU

### 🟨 Square (`N = K`)

| Source shape (M, N, K) | GELU matrix | Model | Paper / source |
|---|---|---|---|
| M, 3072, 3072 | `C[M, 3072]` | Phi-4-mini | [Phi-4-Mini Technical Report](https://arxiv.org/abs/2503.01743) |
| M, 3072, 3072 | `C[M, 3072]` | Phi-4-mini | [Phi-4-Mini Technical Report](https://arxiv.org/abs/2503.01743) |
| M, 5120, 5120 | `C[M, 5120]` | Llama 4 Scout | [Model source](https://www.llama.com/models/llama-4/) |
| M, 5120, 5120 | `C[M, 5120]` | Llama 4 Scout | [Model source](https://www.llama.com/models/llama-4/) |
| M, 2880, 2880 | `C[M, 2880]` | GPT-OSS-120B | [GPT-OSS Model Card](https://arxiv.org/abs/2508.10925) |

### 🟩 Tall and thin (`N > K`)

| Source shape (M, N, K) | GELU matrix | Model | Paper / source |
|---|---|---|---|
| M, 6144, 5120 | `C[M, 6144]` | Qwen3.8-27B | [Qwen3.8 model source](https://huggingface.co/Qwen/Qwen3.8-27B) |
| M, 17408, 5120 | `C[M, 17408]` | Qwen3.8-27B | [Qwen3.8 model source](https://huggingface.co/Qwen/Qwen3.8-27B) |
| M, 21504, 5376 | `C[M, 21504]` | Gemma 4 31B | [Gemma 4 Technical Report](https://arxiv.org/abs/2607.02770) |
| M, 8192, 5120 | `C[M, 8192]` | Llama 4 Scout | [Model source](https://www.llama.com/models/llama-4/) |
| M, 262144, 5376 | `C[M, 262144]` | Gemma 4 31B | [Gemma 4 Technical Report](https://arxiv.org/abs/2607.02770) |

### 🟦 Short and fat (`K > N`)

| Source shape (M, N, K) | GELU matrix | Model | Paper / source |
|---|---|---|---|
| M, 1024, 5120 | `C[M, 1024]` | Qwen3.8-27B | [Qwen3.8 model source](https://huggingface.co/Qwen/Qwen3.8-27B) |
| M, 4096, 5376 | `C[M, 4096]` | Gemma 4 31B | [Gemma 4 Technical Report](https://arxiv.org/abs/2607.02770) |
| M, 512, 2880 | `C[M, 512]` | GPT-OSS-120B | [GPT-OSS Model Card](https://arxiv.org/abs/2508.10925) |
| M, 2304, 5120 | `C[M, 2304]` | DeepSeek V4.1-Flash-1017 | [DeepSeek-V4.1-Flash Technical Report](https://arxiv.org/abs/2609.19969) |
| M, 5376, 21504 | `C[M, 5376]` | Gemma 4 31B | [Gemma 4 Technical Report](https://arxiv.org/abs/2607.02770) |