#!/bin/bash

set -e

ARCH=$1

LAlib=$2
LAname=$3
LAinc=$4

MPlib=$5
MPname=$6
MPinc=$7

HPL_PATH=${8:-'hpl'}
CALLERdir="$(pwd)"
TOPdir="$(pwd)/${HPL_PATH}"

if [[ -z "$ARCH" || -z "$LAlib" || -z "$LAinc" || -z "$MPlib" || -z "$MPname" || -z "$MPinc" ]]; then
  echo "Error: Missing required arguments."
  echo "Usage: $0 <ARCH> <LAlib> <LAname> <LAinc> <MPlib> <MPname> <MPinc> [HPL_PATH]"
  exit 1
fi

sed "s|#ARCH#|${ARCH}|g; s|#LAlib#|${LAlib}|g; s|#LAname#|${LAname}|g; s|#LAinc#|${LAinc}|g; s|#MPlib#|${MPlib}|g; s|#MPname#|${MPname}|g; s|#MPinc#|${MPinc}|g; s|#TOPdir#|${TOPdir}|g" Make.in > Make.out

if [[ ! -d ${HPL_PATH} ]]; then
  curl -O https://www.netlib.org/benchmark/hpl/hpl-2.3.tar.gz
  tar -xzf hpl-2.3.tar.gz
  rm hpl-2.3.tar.gz
  mv hpl-2.3 ${HPL_PATH}
else
  echo "Directory ${HPL_PATH} already exists; assuming source code has been downloaded before"
fi

cd ${HPL_PATH}
cp ../Make.out "./Make.${ARCH}"

if [ -d bin ]; then
  echo "directory \"bin\" already exists - deleting previous HPL build"
  rm -rf bin;
fi

# ./configure
make -j16 "arch=${ARCH}"
cp "bin/${ARCH}/xhpl" ..

if [[ -z $? ]]; then
  echo "Could'n find the xhpl binary"
  exit 1
fi

echo "Done!"
echo "xhpl binary available in '${CALLERdir}'"