#!/bin/bash

## !! Please run this from a Pioneer board !!

set -e

module load llvm/EPI-development

cp Makefile STREAM/
cd STREAM

# sed -i 's/CC := gcc/CC := clang/' Makefile
# Preprocessor variables do not work with clang compiler
sed -E -i 's/(define STREAM_ARRAY_SIZE)[[:space:]]+[0-9]+/\1 20000000/' stream.c

sleep 2

make clean
CC=clang make stream_c_custom