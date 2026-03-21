# High Performance Conjugate Gradient

## Build

<!-- ### CMake

```bash
# Make sure to have modules/paths setup correctly
# Internally uses git and CMake
./download_and_build_cmake.sh
``` -->

### Makefile

<!-- ```bash
# Make sure to have modules/paths setup correctly
# Internally uses git and Makefile
# Example: Architecture, Path to Linear Algebra lib, Path to MPI
./download_and_build_makefile.sh aarch64 \
    /path/to/OpenBLAS/lib/libopenblas.a openblas /path/to/OpenBLAS/include \
    /path/to/MPI/lib/libopenmpi.a openmpi /path/to/MPI/include
``` -->

Examples

```bash
# HAICGU
./build_makefile.sh kp920 /software/kp920-RL8/Stages/2024/software/OpenMPI/5.0.3-GCC-14.1.0/lib64 mpi /software/kp920-RL8/Stages/2024/software/OpenMPI/5.0.3-GCC-14.1.0/include

# BSC-HCA
./build_makefile.sh riscv /apps/riscv/openmpi/4.1.6_llvm1.0_native+cross/lib mpi /apps/riscv/openmpi/4.1.6_llvm1.0_native+cross/include
```

## Run

```bash
# Assuming you set up SbatchMan aliases 
sbmi
sbmc -f configs.yaml -ow
sbml -f jobs.yaml
```

## Generate Data

Once all experiments are done:

```bash
python3 parse_results.py
```

## Plots with data from Multiple Sources

1) Sync the `hpcg_results_<cluster>.parquet` files on one machine
2) Run:
```bash
python3 plots.py hpcg_results_<cluster1>.parquet hpcg_results_<cluster2>.parquet ...
```