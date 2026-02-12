# High-Performance LINPACK (CPU-only)

Before running `compile.sh` make sure that you have:
- gcc
- mpicc

To download and compile the official benchmark use `compile.sh`, providing the proper arguments.

For instance:
```bash
# Leonardo
ml gcc/12.2.0 nvhpc/24.5 hpcx-mpi/2.19 openblas/0.3.26--nvhpc--24.5 
./compile.sh x86_64 /leonardo/prod/spack/06/install/0.22/linux-rhel8-icelake/nvhpc-24.5/openblas-0.3.26-w3alzpga4fq43or2nu7763np2dpmiipt/lib openblas /leonardo/prod/spack/06/install/0.22/linux-rhel8-icelake/nvhpc-24.5/openblas-0.3.26-w3alzpga4fq43or2nu7763np2dpmiipt/include /leonardo/prod/spack/06/install/0.22/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.5-torlmnyzcexnrs6pq4cccabv7ehkv3xy/Linux_x86_64/24.5/comm_libs/12.4/hpcx/hpcx-2.19/ompi/lib mpi /leonardo/prod/spack/06/install/0.22/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.5-torlmnyzcexnrs6pq4cccabv7ehkv3xy/Linux_x86_64/24.5/comm_libs/12.4/hpcx/hpcx-2.19/ompi/include
```