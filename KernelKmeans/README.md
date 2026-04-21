# Popcorn (Kernel K-Means)

TODOs:
* Kernels runtime breakdown
* Write better output
* Restructure kernels
* Experiments with SbatchMan
* Parsing and plots

## Example Manual Run

```bash
export OMP_NUM_THREADS=4
./build_bsc-hca_clang/popcornkmeans_openmp -n 1000 -d 3 -k 15 -m 10 --init random -f linear -l 2 --runs 5
```

<!-- ## OpenBLAS manual compilation on BSC HCA Cluster

```c
// test_blas.c
#include <cblas.h>
#include <stdio.h>

int main()
{
  int i=0;
  double A[6] = {1.0,2.0,1.0,-3.0,4.0,-1.0};
  double B[6] = {1.0,2.0,1.0,-3.0,4.0,-1.0};
  double C[9] = {.5,.5,.5,.5,.5,.5,.5,.5,.5};
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,3,3,2,1,A, 3, B, 3,2,C,3);

  for(i=0; i<9; i++)
    printf("%lf ", C[i]);
  printf("\n");
  return 0;
}
```

```bash
# GCC
gcc -o test_blas_gcc test_blas.c ${OBLAS_INCL} ${OBLAS_LIBS} -lopenblas

# LLVM (clang)
clang -o test_blas_clang test_blas.c ${OBLAS_INCL} ${OBLAS_LIBS} -lopenblas -L/apps/riscv/llvm/EPI/development/lib/riscv64-unknown-linux-gnu -lflang_rt.runtime -lm
``` -->