#!/bin/bash

cd

wget https://ssh.hca.bsc.es/epi/ftp/llvm-EPI-development-toolchain-native-latest.tar.bz2 #for RVV 1.0
wget https://ssh.hca.bsc.es/epi/ftp/llvm-EPI-0.7-development-toolchain-native-latest.tar.bz2 #for RVV 0.7.1

mkdir software_env

tar -xf llvm-EPI-development-toolchain-native-latest.tar.bz2
mv llvm-EPI-development-toolchain-native software_env/llvm-EPI-rvv1
rm llvm-EPI-development-toolchain-native-latest.tar.bz2

tar -xf llvm-EPI-0.7-development-toolchain-native-latest.tar.bz2
mv llvm-EPI-0.7-development-toolchain-native software_env/llvm-EPI-rvv071
rm llvm-EPI-0.7-development-toolchain-native-latest.tar.bz2
