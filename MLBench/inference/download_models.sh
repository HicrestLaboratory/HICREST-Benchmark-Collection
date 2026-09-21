#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

MODELS_DIR="models"
mkdir -p "$MODELS_DIR"

download() {
    local url="$1" dest="$MODELS_DIR/$2"
    if [[ -f "$dest" ]]; then
        echo "[skip] $2"
        return
    fi
    echo "[fetch] $2"
    curl -fL --retry 5 --retry-delay 3 -C - -o "$dest.part" "$url"
    mv "$dest.part" "$dest"
}

download "https://huggingface.co/lmstudio-community/granite-4.0-h-micro-GGUF/resolve/main/granite-4.0-h-micro-Q4_K_M.gguf" \
         "granite-4.0-h-micro-Q4_K_M.gguf"

download "https://huggingface.co/waqar-hpe/mamba-2.8b-hf-Q4_K_M-GGUF/resolve/main/mamba-2.8b-hf-q4_k_m.gguf" \
         "mamba-2.8b-hf-q4_k_m.gguf"

download "https://huggingface.co/bartowski/google_gemma-4-E2B-it-GGUF/resolve/main/google_gemma-4-E2B-it-Q4_K_M.gguf" \
         "google_gemma-4-E2B-it-Q4_K_M.gguf"

download "https://huggingface.co/bartowski/granite-3.1-3b-a800m-instruct-GGUF/resolve/main/granite-3.1-3b-a800m-instruct-Q4_K_M.gguf" \
         "granite-3.1-3b-a800m-instruct-Q4_K_M.gguf"

ls -lh "$MODELS_DIR"