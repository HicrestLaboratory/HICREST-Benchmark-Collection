# NVIDIA High-Performance LINPACK

To download and compile NVIDIA's official benchmark use `compile.sh`, providing the proper arguments.

```
./compile.sh -h
Usage: ./compile.sh -a <arch> -m <mpi> [-v <version>]

  -a  Architecture: x86_64 | arm64-sbsa
  -m  MPI implementation: openmpi | mpich
  -v  Version (default: 25.09.06)

Example:
  ./compile.sh -a x86_64 -m openmpi
```