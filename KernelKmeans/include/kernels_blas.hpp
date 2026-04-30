#ifndef KERNELS_OPENBLAS_H
#define KERNELS_OPENBLAS_H

#include <bits/stdc++.h>
#include <omp.h>
#include <ccutils/colors.h>

#if BLA_VENDOR == 1
  #include <nvpl_blas_cblas.h>
// #if BLA_VENDOR == 2 MKL TODO
#else
  #include <cblas.h>
#endif

/**
 * ============================================================================
 * POPCORN KERNEL K-MEANS: OpenBLAS IMPLEMENTATION
 * ============================================================================
 * 
 * Paper: Popcorn: Accelerating Kernel K-means on GPUs through Sparse Linear Algebra
 * Conference: PPoPP '25
 * 
 * This implementation uses OpenBLAS (CBLAS) for all dense matrix operations
 * and properly implements Equation 10 from the paper:
 * 
 *   D = -2·K·V^T + P̃ + C̃
 * 
 * Where:
 *   K ∈ ℝ^(n×n)  - kernel matrix
 *   V ∈ ℝ^(k×n)  - sparse cluster membership matrix (CSR format)
 *   P̃ ∈ ℝ^(n×k)  - point norms (P̃_{i,j} = ||φ(p_i)||²)
 *   C̃ ∈ ℝ^(n×k)  - centroid norms (C̃_{i,j} = ||c_j||²)
 *   D ∈ ℝ^(n×k)  - output distances (row-major)
 * ============================================================================
 */

/**
 * @brief Finds the closest centroid for each point
 * 
 * For each point, finds the cluster with minimum distance.
 * Thread-safe cluster counting using atomic operations.
 *
 * @param n number of points
 * @param k number of clusters
 * @param distances D matrix (n×k, row-major)
 * @param points_clusters output: cluster assignment for each point
 * @param clusters_len output: count of points in each cluster
 */
void clusters_argmin_cpu(const uint32_t n, const uint32_t k, 
                         const float* distances, uint32_t* points_clusters,  
                         uint32_t* clusters_len) {
  
  std::fill(clusters_len, clusters_len + k, 0);
  
  #pragma omp parallel for default(none) \
          shared(distances, points_clusters, clusters_len, n, k)
  for (uint32_t i = 0; i < n; i++) {
    float min_dist = std::numeric_limits<float>::max();
    uint32_t min_cluster = 0;
    
    for (uint32_t j = 0; j < k; j++) {
      float dist = distances[i * k + j];
      if (dist < min_dist) {
        min_dist = dist;
        min_cluster = j;
      }
    }
    
    points_clusters[i] = min_cluster;
    #pragma omp atomic
    clusters_len[min_cluster]++;
  }
}

/**
 * @brief Compute kernel matrix K = P̂·P̂^T using BLAS GEMM
 * 
 * Uses cblas_sgemm for optimal performance with:
 * - Multi-threaded BLAS library
 * - Cache-efficient blocking
 * - SIMD vectorization
 *
 * @param n number of points
 * @param d dimensionality
 * @param points P̂ matrix (n×d, row-major)
 * @param K output: kernel matrix (n×n, row-major)
 */
void init_kernel_mtx_gemm_cpu(const unsigned long long n,
                              const uint32_t d,
                              const float* points,
                              float* K) {
  // K = P̂ * P̂^T
  cblas_sgemm(CblasRowMajor,        // Row-major layout
              CblasNoTrans,          // A: no transpose
              CblasTrans,            // B: transpose
              n, n, d,               // M, N, K_dim
              1.0f,                  // alpha
              points, d,             // A = P̂ (n×d)
              points, d,             // B = P̂ (n×d)
              0.0f,                  // beta
              K, n);                 // C = K (n×n)
}

/**
 * @brief Apply linear kernel: K = -2·K element-wise using BLAS SCAL
 * 
 * Uses cblas_sscal to multiply all elements efficiently:
 * K[i,j] *= -2.0 for all i,j
 *
 * @param n matrix dimension (n×n)
 * @param K matrix to transform in-place
 */
void apply_linear_kernel_cpu(const uint32_t n, float* K) {
  cblas_sscal(n * n, -2.0f, K, 1);
}

/**
 * @brief Apply polynomial kernel: K = -2·(K + 1)²
 * 
 * Step 1: Shift all elements by 1 using BLAS AXPY
 * Step 2: Square and scale element-wise (OpenMP, element-wise only)
 *
 * @param n matrix dimension
 * @param K matrix to transform in-place
 */
void apply_polynomial_kernel_cpu(const uint32_t n, float* K) {
  // Step 1: K = K + 1 using AXPY
  // y := alpha*x + y
  std::vector<float> ones(n * n, 1.0f);
  cblas_saxpy(n * n, 1.0f, ones.data(), 1, K, 1);
  
  // Step 2: K = -2 * K^2 element-wise
  #pragma omp parallel for default(none) shared(K, n)
  for (uint64_t i = 0; i < (uint64_t)n * n; i++) {
    K[i] = -2.0f * K[i] * K[i];
  }
}

/**
 * @brief Apply sigmoid kernel: K = -2·tanh(K + 1)
 * 
 * Step 1: Shift elements using BLAS AXPY
 * Step 2: Apply tanh and scale element-wise (OpenMP)
 *
 * @param n matrix dimension
 * @param K matrix to transform in-place
 */
void apply_sigmoid_kernel_cpu(const uint32_t n, float* K) {
  // Step 1: K = K + 1 using AXPY
  std::vector<float> ones(n * n, 1.0f);
  cblas_saxpy(n * n, 1.0f, ones.data(), 1, K, 1);
  
  // Step 2: K = -2 * tanh(K) element-wise
  #pragma omp parallel for default(none) shared(K, n)
  for (uint64_t i = 0; i < (uint64_t)n * n; i++) {
    K[i] = -2.0f * std::tanh(K[i]);
  }
}

/**
 * @brief Extract diagonal of K and scale: P̃[i] = K[i,i]
 * 
 * Efficiently copies diagonal elements into vector.
 * Used to get point norms: P̃_{i,j} = ||φ(p_i)||² (same for all j)
 *
 * @param K kernel matrix (n×n)
 * @param p_norms output: diagonal vector (n elements)
 * @param n matrix dimension
 */
void copy_diag_cpu(const float* K, float* p_norms, const uint32_t n) {
  #pragma omp parallel for default(none) shared(K, p_norms, n)
  for (uint32_t i = 0; i < n; i++) {
    p_norms[i] = K[i * n + i];
  }
}

/**
 * @brief Build CSR sparse matrix V (cluster membership) from cluster assignments
 * 
 * V[j,i] = 1/|L_j| if point i is in cluster j, else 0
 * Stored in CSR format: (vals, colinds, rowptrs)
 *
 * @param points_clusters point-to-cluster mapping
 * @param clusters_len cluster sizes
 * @param vals output: non-zero values
 * @param colinds output: column indices
 * @param rowptrs output: row pointers
 * @param n number of points
 * @param k number of clusters
 */
void compute_v_sparse_csr_cpu(const uint32_t* points_clusters,
                              const uint32_t* clusters_len,
                              float* vals,
                              int32_t* colinds,
                              int32_t* rowptrs,
                              const uint32_t n,
                              const uint32_t k) {
  
  // Count non-zeros per row
  std::vector<uint32_t> row_counts(k, 0);
  for (uint32_t i = 0; i < n; i++) {
    row_counts[points_clusters[i]]++;
  }
  
  // Compute row offsets
  rowptrs[0] = 0;
  for (uint32_t j = 0; j < k; j++) {
    rowptrs[j + 1] = rowptrs[j] + row_counts[j];
  }
  
  // Fill CSR arrays
  std::vector<uint32_t> current_pos(rowptrs, rowptrs + k);
  for (uint32_t i = 0; i < n; i++) {
    uint32_t cluster = points_clusters[i];
    uint32_t pos = current_pos[cluster]++;
    vals[pos] = 1.0f / clusters_len[cluster];
    colinds[pos] = i;
  }
}

/**
 * @brief Compute distances using complete formula: D = -2·K·V^T + P̃ + C̃
 * 
 * Implements Equation 10 from paper with three steps:
 * 1. SpMM: E = -2·K·V^T (sparse-dense multiplication)
 * 2. Extract z[i] = E[cluster[i], i] for centroid norm computation
 * 3. SpMV: c_norms = -0.5·V·z
 * 4. Final: D[i,j] = E[i,j] + P̃[i] + C̃[j]
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
                                    float* distances) {
  
  // Step 1: Compute E = -2·K·V^T (SpMM)
  // V is k×n (sparse), K is n×n (dense), E is k×n (dense)
  std::vector<float> E(k * n, 0.0f);
  
  #pragma omp parallel for default(none) \
          shared(E, K, V_vals, V_colinds, V_rowptrs, n, k)
  for (uint32_t j = 0; j < k; j++) {
    // For cluster j (sparse row), compute E[j,:] = -2·V[j,:]·K
    for (uint32_t i = 0; i < n; i++) {
      float sum = 0.0f;
      // V[j,:] · K[:,i] = sum over ℓ (V[j,ℓ]·K[ℓ,i])
      for (int32_t idx = V_rowptrs[j]; idx < V_rowptrs[j + 1]; idx++) {
        int32_t col = V_colinds[idx];
        sum += V_vals[idx] * K[col * n + i];
      }
      E[j * n + i] = -2.0f * sum;
    }
  }
  
  // Step 2: Extract z[i] = E[cluster[i], i]
  std::vector<float> z(n);
  #pragma omp parallel for default(none) \
          shared(z, E, points_clusters, n, k)
  for (uint32_t i = 0; i < n; i++) {
    uint32_t cluster = points_clusters[i];
    z[i] = E[cluster * n + i];
  }
  
  // Step 3: Compute c_norms = -0.5·V·z (SpMV)
  std::vector<float> c_norms(k, 0.0f);
  #pragma omp parallel for default(none) \
          shared(c_norms, V_vals, V_colinds, V_rowptrs, z, k)
  for (uint32_t j = 0; j < k; j++) {
    float sum = 0.0f;
    for (int32_t idx = V_rowptrs[j]; idx < V_rowptrs[j + 1]; idx++) {
      int32_t col = V_colinds[idx];
      sum += V_vals[idx] * z[col];
    }
    c_norms[j] = -0.5f * sum;
  }
  
  // Step 4: Assemble D = E^T + P̃ + C̃
  // Since E is k×n and we want n×k output, we need to transpose
  // D[i,j] = E[j,i] + p_norms[i] + c_norms[j]
  
  #pragma omp parallel for collapse(2) default(none) \
          shared(distances, E, p_norms, c_norms, n, k)
  for (uint32_t i = 0; i < n; i++) {
    for (uint32_t j = 0; j < k; j++) {
      distances[i * k + j] = E[j * n + i] + p_norms[i] + c_norms[j];
    }
  }
}

/**
 * @brief Compute distances using BLAS GERC for final assembly (alternative)
 * 
 * More BLAS-optimized version using rank-1 updates:
 * 1. Compute E and c_norms as above
 * 2. Use BLAS GERC to add outer products efficiently
 *
 * Note: This is more BLAS-centric but usually not faster due to memory layout
 */
void compute_distances_gerc_cpu(const uint32_t n,
                               const uint32_t k,
                               const float* K,
                               const float* p_norms,
                               const float* V_vals,
                               const int32_t* V_colinds,
                               const int32_t* V_rowptrs,
                               const uint32_t* points_clusters,
                               float* distances) {
  
  // Step 1: Compute E = -2·K·V^T
  std::vector<float> E(k * n, 0.0f);
  
  #pragma omp parallel for default(none) \
          shared(E, K, V_vals, V_colinds, V_rowptrs, n, k)
  for (uint32_t j = 0; j < k; j++) {
    for (uint32_t i = 0; i < n; i++) {
      float sum = 0.0f;
      for (int32_t idx = V_rowptrs[j]; idx < V_rowptrs[j + 1]; idx++) {
        int32_t col = V_colinds[idx];
        sum += V_vals[idx] * K[col * n + i];
      }
      E[j * n + i] = -2.0f * sum;
    }
  }
  
  // Step 2: Extract z and compute c_norms
  std::vector<float> z(n);
  #pragma omp parallel for default(none) \
          shared(z, E, points_clusters, n, k)
  for (uint32_t i = 0; i < n; i++) {
    uint32_t cluster = points_clusters[i];
    z[i] = E[cluster * n + i];
  }
  
  std::vector<float> c_norms(k, 0.0f);
  #pragma omp parallel for default(none) \
          shared(c_norms, V_vals, V_colinds, V_rowptrs, z, k)
  for (uint32_t j = 0; j < k; j++) {
    float sum = 0.0f;
    for (int32_t idx = V_rowptrs[j]; idx < V_rowptrs[j + 1]; idx++) {
      int32_t col = V_colinds[idx];
      sum += V_vals[idx] * z[col];
    }
    c_norms[j] = -0.5f * sum;
  }
  
  // Step 3: Transpose E to get E_T (n×k)
  std::vector<float> E_T(n * k);
  #pragma omp parallel for collapse(2) default(none) \
          shared(E_T, E, n, k)
  for (uint32_t i = 0; i < n; i++) {
    for (uint32_t j = 0; j < k; j++) {
      E_T[i * k + j] = E[j * n + i];
    }
  }
  
  // Step 4: D = E_T + ones·c_norms^T + p_norms·ones^T (using BLAS GERC)
  std::memcpy(distances, E_T.data(), n * k * sizeof(float));
  
  // Add c_norms using rank-1 update: D += ones·c_norms^T
  std::vector<float> ones(n, 1.0f);
  cblas_sger(CblasRowMajor,           // Row-major layout
             n, k,                    // M, N (result is n×k)
             1.0f,                    // alpha
             ones.data(), 1,          // x = ones (n elements, stride 1)
             c_norms.data(), 1,       // y = c_norms (k elements, stride 1)
             distances, k);           // A = distances (n×k, lda=k)
  
  // Add p_norms using rank-1 update: D += p_norms·ones^T
  cblas_sger(CblasRowMajor,
             n, k,
             1.0f,
             (float*)p_norms, 1,      // x = p_norms (const pointer)
             ones.data(), 1,          // y = ones
             distances, k);
}

#endif // KERNELS_OPENBLAS_H