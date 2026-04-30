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

// ============================================================================
// IMPLEMENTATION SELECTION
// ============================================================================
// Include the actual implementation (OpenBLAS or OpenMP)

#if defined(BLA_VENDOR)
  #include "kernels_blas.hpp"
#else
  // Default: OpenMP (no external dependencies)
  #include "kernels_openmp.hpp"
#endif

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

#endif // __KERNELS_H__
