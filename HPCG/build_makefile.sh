#!/bin/bash

set -e

mkdir -p hpcg-out

ARCH=$1

MPlib=$2
MPname=$3
MPinc=$4

HPCG_PATH=${8:-'hpcg-cpu'}
TOPdir="$(pwd)/${HPCG_PATH}"

if [[ -z "$ARCH" || -z "$MPlib" || -z "$MPname" || -z "$MPinc" ]]; then
  echo "Error: Missing required arguments."
  echo "Usage: $0 <ARCH> <MPlib> <MPname> <MPinc> [HPCG_PATH]"
  exit 1
fi

sed "s|#ARCH#|${ARCH}|g; s|#MPlib#|${MPlib}|g; s|#MPname#|${MPname}|g; s|#MPinc#|${MPinc}|g; s|#TOPdir#|${TOPdir}|g" Make.in > Make.out

if [[ ! -d ${HPCG_PATH} ]]; then
  echo "Directory ${HPCG_PATH} does not exists. This should not happen!"
  exit 1
fi

cd ${HPCG_PATH}
mkdir -p bin
cp ../Make.out "./setup/Make.${ARCH}"

# Makefile build
make -j16 "arch=${ARCH}"

cp ./bin/xhpcg ..

echo "Done!"
