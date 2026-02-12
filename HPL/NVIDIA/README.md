# NVIDIA High-Performance LINPACK

Before running `compile.sh` make sure that you have:
- nvcc
- mpicc

To download and compile NVIDIA's official benchmark use `compile.sh`, providing the proper arguments.

```
./compile.sh -h
Usage: ./compile.sh -a <arch> -m <mpi> [-v <version>] [-cuda <version>]

  -a     Architecture: x86_64 | arm64-sbsa
  -m     MPI implementation: openmpi | mpich
  -v     Benchmark Version (default: 25.09.06)
  -cuda  CUDA Version 12 | 13 (default: 12)

Example:
  ./compile.sh -a x86_64 -m openmpi
```

For instance:
```bash
# Leonardo
# ml gcc/12.2.0 cuda/12.2 openmpi/4.1.6--gcc--12.2.0-cuda-12.2
ml gcc/12.2.0 nvhpc/24.5 hpcx-mpi/2.19
./compile.sh -a x86_64 -m openmpi
```