#!/bin/bash

## !! Please run this from a Pioneer board !!

module load llvm/EPI-development

cd pointer-chasing

sed -i 's/CXX := g++/CXX := clang++/' Makefile
make clean
make