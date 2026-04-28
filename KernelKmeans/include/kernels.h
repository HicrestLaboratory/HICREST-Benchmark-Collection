#ifndef __KERNELS_H__
#define __KERNELS_H__

#include <stdint.h>
#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>
#include <cmath>
#include <omp.h>

#include "common.h"

/**
 * ============================================================================
 * UNIFIED KERNEL INTERFACE FOR POPCORN KERNEL K-MEANS
 * ============================================================================
 * 
 * This header provides a unified interface that works with both:
 * - OpenBLAS implementation (optimized matrix operations)
 * - OpenMP implementation (pure parallel loops)
 * 
 * To switch implementations, define either:
 *   #define USE_OPENBLAS    // Uses CBLAS for dense operations
 * OR
 *   #define USE_OPENMP      // Uses pure OpenMP (default)
 * 
 * The API is identical - only the #include changes!
 * ============================================================================
 */

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

// ============================================================================
// KERNEL TRANSFORMATION FUNCTORS
// ============================================================================

/**
 * @brief Linear kernel transformation: K *= -2.0
 */
struct LinearKernel {
  static void function(const uint32_t n, float* K);
};

/**
 * @brief Polynomial kernel: K = -2*(K+1)^2
 */
struct PolynomialKernel {
  static void function(const uint32_t n, float* K);
};

/**
 * @brief Sigmoid kernel: K = -2*tanh(K+1)
 */
struct SigmoidKernel {
  static void function(const uint32_t n, float* K);
};

// ============================================================================
// UNIFIED API - THESE FUNCTIONS WORK WITH BOTH IMPLEMENTATIONS
// ============================================================================

/**
 * @brief Find closest cluster for each point
 * 
 * For each point, finds cluster with minimum distance and updates
 * cluster assignments and cluster sizes atomically.
 * 
 * @param n number of points
 * @param k number of clusters
 * @param distances D matrix (n×k, row-major)
 * @param points_clusters output: cluster ID for each point
 * @param clusters_len output: count of points in each cluster
 */
void clusters_argmin_cpu(const uint32_t n, const uint32_t k, 
                         const float* distances, uint32_t* points_clusters,  
                         uint32_t* clusters_len);

/**
 * @brief Compute kernel matrix K = P̂·P̂^T
 * 
 * Uses optimized dense matrix multiplication (BLAS GEMM on OpenBLAS,
 * manual GEMM with collapse(2) on OpenMP).
 *
 * @param n number of points
 * @param d dimensionality
 * @param points P̂ matrix (n×d, row-major)
 * @param K output: kernel matrix (n×n, row-major)
 */
void init_kernel_mtx_gemm_cpu(const unsigned long long n,
                              const uint32_t d,
                              const float* points,
                              float* K);

/**
 * @brief Apply linear kernel transformation: K *= -2.0
 * 
 * @param n matrix dimension
 * @param K matrix to transform in-place
 */
void apply_linear_kernel_cpu(const uint32_t n, float* K);

/**
 * @brief Apply polynomial kernel: K = -2*(K+1)^2
 * 
 * @param n matrix dimension
 * @param K matrix to transform in-place
 */
void apply_polynomial_kernel_cpu(const uint32_t n, float* K);

/**
 * @brief Apply sigmoid kernel: K = -2*tanh(K+1)
 * 
 * @param n matrix dimension
 * @param K matrix to transform in-place
 */
void apply_sigmoid_kernel_cpu(const uint32_t n, float* K);

/**
 * @brief Extract diagonal of K: P̃[i] = K[i,i]
 * 
 * Copies diagonal elements for point norm computation.
 * These are used to build the P̃ matrix in the distance formula.
 *
 * @param K kernel matrix (n×n)
 * @param p_norms output: diagonal vector (n elements)
 * @param n matrix dimension
 */
void copy_diag_cpu(const float* K, float* p_norms, const uint32_t n);

/**
 * @brief Build sparse matrix V in CSR format
 * 
 * Creates cluster membership matrix from cluster assignments.
 * V[j,i] = 1/|L_j| if point i is in cluster j, 0 otherwise
 *
 * @param points_clusters cluster assignment for each point
 * @param clusters_len cluster sizes
 * @param vals output: CSR values (n elements)
 * @param colinds output: CSR column indices (n elements)
 * @param rowptrs output: CSR row pointers (k+1 elements)
 * @param n number of points
 * @param k number of clusters
 */
void compute_v_sparse_csr_cpu(const uint32_t* points_clusters,
                              const uint32_t* clusters_len,
                              float* vals,
                              int32_t* colinds,
                              int32_t* rowptrs,
                              const uint32_t n,
                              const uint32_t k);

/**
 * @brief Compute complete distance matrix: D = -2·K·V^T + P̃ + C̃
 * 
 * Implements Equation 10 from Popcorn paper with full formula including
 * point norms and centroid norms. This is the main computational kernel.
 * 
 * Algorithm:
 * 1. SpMM: E = -2·K·V^T (sparse-dense multiplication)
 * 2. Extract: z[i] = E[cluster[i], i]
 * 3. SpMV: c_norms = -0.5·V·z
 * 4. Assemble: D[i,j] = E[j,i] + p_norms[i] + c_norms[j]
 *
 * @param n number of points
 * @param k number of clusters
 * @param K kernel matrix (n×n, row-major)
 * @param p_norms point norms (n elements)
 * @param V_vals CSR values
 * @param V_colinds CSR column indices
 * @param V_rowptrs CSR row pointers
 * @param points_clusters cluster assignments
 * @param distances output: D matrix (n×k, row-major)
 */
void compute_distances_complete_cpu(const uint32_t n,
                                    const uint32_t k,
                                    const float* K,
                                    const float* p_norms,
                                    const float* V_vals,
                                    const int32_t* V_colinds,
                                    const int32_t* V_rowptrs,
                                    const uint32_t* points_clusters,
                                    float* distances);

// ============================================================================
// LEGACY COMPATIBILITY FUNCTIONS (for backwards compatibility)
// ============================================================================
// These are deprecated but kept for API compatibility with your existing code

/**
 * @deprecated Use compute_distances_complete_cpu instead
 * 
 * Old sparse-dense multiplication. Use the new function which includes
 * the complete formula with P̃ and C̃ terms.
 */
void compute_distances_spmm_cpu(const uint32_t n, const uint32_t k,
                               const float* B,
                               const float* V_vals,
                               const int32_t* V_colinds,
                               const int32_t* V_rowptrs,
                               float* distances);

/**
 * @deprecated Use copy_diag_cpu instead
 * 
 * Extract diagonal with scaling factor.
 */
void copy_diag_scal_cpu(const float* M, float* output,
                        const int m, const int n,
                        const float alpha);

// ============================================================================
// TEMPLATE UTILITIES
// ============================================================================

/**
 * @brief Generic kernel matrix initialization with kernel transformation
 * 
 * Computes kernel matrix and applies kernel transformation in one call.
 * Uses static polymorphism with Kernel functor template parameter.
 *
 * @tparam Kernel kernel transformation functor (LinearKernel, etc.)
 */
template <typename Kernel>
void init_kernel_mtx_cpu(const unsigned long long n,
                         const uint32_t d,
                         const float* points,
                         float* K) {
  // Compute kernel matrix K = P̂·P̂^T
  init_kernel_mtx_gemm_cpu(n, d, points, K);
  
  // Apply kernel transformation
  Kernel::function(n, K);
}

// ============================================================================
// KERNEL TRANSFORMATION IMPLEMENTATIONS
// ============================================================================

/**
 * @brief Linear kernel implementation
 */
inline void LinearKernel::function(const uint32_t n, float* K) {
  apply_linear_kernel_cpu(n, K);
}

/**
 * @brief Polynomial kernel implementation
 */
inline void PolynomialKernel::function(const uint32_t n, float* K) {
  apply_polynomial_kernel_cpu(n, K);
}

/**
 * @brief Sigmoid kernel implementation
 */
inline void SigmoidKernel::function(const uint32_t n, float* K) {
  apply_sigmoid_kernel_cpu(n, K);
}

// ============================================================================
// IMPLEMENTATION SELECTION
// ============================================================================
// Include the actual implementation (OpenBLAS or OpenMP)

#if defined(USE_OPENBLAS)
  #include "kernels_openblas.hpp"
#else
  // Default: OpenMP (no external dependencies)
  #include "kernels_openmp.hpp"
#endif

#endif // __KERNELS_H__
