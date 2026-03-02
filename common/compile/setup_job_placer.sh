#!/bin/bash

echo "==== Setting up JobPlacer ===="

if ! [ -x "$(command -v cargo)" ]; then
    echo "This requires a Rust compiler. If you don't have it, install it locally running:"
    echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
fi
[[ -d "../common/JobPlacer" ]] || { echo "Error: JobPlacer not found. Please get the submodule (from the repo root: \`git submodule update --init common/JobPlacer\`)."; exit 1; }

original_dir=$(pwd)
cd ../common/JobPlacer

cargo build --release
echo "JobPlacer Ready!"

cd "$original_dir"