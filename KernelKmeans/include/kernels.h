#ifndef __KERNEL_CPU_H__
#define __KERNEL_CPU_H__

#include <stdint.h>
#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>
#include <cmath>

#include "common.h"

// CPU equivalents of CUDA structures
struct Pair {
  float v;
  uint32_t i;
};

struct Kvpair {
  uint32_t key;
  uint32_t value;
};

enum class KernelMtxMethod {
  KERNEL_MTX_NAIVE,
  KERNEL_MTX_GEMM,
  KERNEL_MTX_SYRK
};

// Kernel function pointers
struct LinearKernel {
  static void function(const uint32_t n, const uint32_t d, DATA_TYPE* B);
};

struct PolynomialKernel {
  static void function(const uint32_t n, const uint32_t d, DATA_TYPE* B);
};

struct SigmoidKernel {
  static void function(const uint32_t n, const uint32_t d, DATA_TYPE* B);
};

// CPU function declarations
void clusters_argmin_cpu(const uint32_t n, const uint32_t k, 
                        DATA_TYPE* distances, uint32_t* points_clusters,  
                        uint32_t* clusters_len, bool is_row_major);

void compute_centroids_cpu(DATA_TYPE* centroids, const DATA_TYPE* points, 
                          const uint32_t* points_clusters, 
                          const uint32_t* clusters_len, 
                          const uint64_t n, const uint32_t d, 
                          const uint32_t k);

void compute_v_sparse_csr_cpu(DATA_TYPE* vals,
                             int32_t* colinds,
                             int32_t* row_offsets,
                             const uint32_t* points_clusters,
                             const uint32_t* clusters_len,
                             const size_t n,
                             const uint32_t k);

void compute_kernel_matrix_cpu(DATA_TYPE* K, 
                              const DATA_TYPE* P, 
                              const unsigned long long n, 
                              const uint32_t d);

void init_kernel_mtx_gemm_cpu(const unsigned long long n,
                             const uint32_t d,
                             const DATA_TYPE* points,
                             DATA_TYPE* B);

void apply_linear_kernel_cpu(const uint32_t n, const uint32_t d, DATA_TYPE* B);
void apply_polynomial_kernel_cpu(const uint32_t n, const uint32_t d, DATA_TYPE* B);
void apply_sigmoid_kernel_cpu(const uint32_t n, const uint32_t d, DATA_TYPE* B);

void copy_diag_scal_cpu(const DATA_TYPE* M, DATA_TYPE* output,
                       const int m, const int n,
                       const DATA_TYPE alpha);

void compute_distances_spmm_cpu(const uint32_t n, const uint32_t k,
                               const DATA_TYPE* B,
                               const DATA_TYPE* V_vals,
                               const int32_t* V_colinds,
                               const int32_t* V_rowptrs,
                               DATA_TYPE* distances);

void sum_points_cpu(const DATA_TYPE* K,
                   const int32_t* clusters,
                   const uint32_t* clusters_len,
                   DATA_TYPE* distances,
                   const uint32_t n, const uint32_t k);

void sum_centroids_cpu(const DATA_TYPE* tmp,
                      const int32_t* clusters,
                      const uint32_t* clusters_len,
                      DATA_TYPE* centroids,
                      const uint32_t n, const uint32_t k);

void compute_distances_naive_cpu(const DATA_TYPE* K,
                                const DATA_TYPE* centroids,
                                const DATA_TYPE* tmp,
                                DATA_TYPE* distances,
                                const uint32_t n, const uint32_t k);

// Template kernel initialization
template <typename Kernel>
void init_kernel_mtx_cpu(const unsigned long long n,
                        const uint32_t k,
                        const uint32_t d,
                        const DATA_TYPE* points,
                        DATA_TYPE* B) {
  // Use GEMM for kernel matrix computation
  init_kernel_mtx_gemm_cpu(n, d, points, B);
  
  // Apply kernel transformation
  Kernel::function(n, d, B);
}

// Kernel function implementations for static polymorphism
inline void LinearKernel::function(const uint32_t n, const uint32_t d, DATA_TYPE* B) {
  apply_linear_kernel_cpu(n, d, B);
}

inline void PolynomialKernel::function(const uint32_t n, const uint32_t d, DATA_TYPE* B) {
  apply_polynomial_kernel_cpu(n, d, B);
}

inline void SigmoidKernel::function(const uint32_t n, const uint32_t d, DATA_TYPE* B) {
  apply_sigmoid_kernel_cpu(n, d, B);
}

#endif // __KERNEL_CPU_H__