#!/bin/bash

set -e

cp Makefile STREAM/
cd STREAM

sed -E -i 's/(define STREAM_ARRAY_SIZE)[[:space:]]+[0-9]+/\1 20000000/' stream.c

sleep 2

make clean
make stream_c_custom