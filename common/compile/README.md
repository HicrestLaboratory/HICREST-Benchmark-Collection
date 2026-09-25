## Setting up the EPI LLVM (clang/clang++) compiler

```bash
chmod +x install_epi_llvm.sh
# Can be run from the login node
./install_epi_llvm.sh
```

This will install the compiler binaries in:
- `~/llvm-EPI-rvv1/bin/clang` (and `clang++`)
- `~/llvm-EPI-rvv071/bin/clang` (and `clang++`)

**The compiler must be invoked from a RISC-V system (no cross-compilation).**


## Setting up the GCC and CLANG

```bash
chmod +x install_compilers.sh
# Can be run from the login node

# Installs {GCC, CLANG} x {x86, arm, riscv64}
./install_compilers.sh --all

# Installs {GCC, CLANG} x {riscv64}
./install_compilers.sh --all --isa riscv64
```