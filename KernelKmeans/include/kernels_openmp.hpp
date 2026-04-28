#pragma once

#include <bits/stdc++.h>
#include <omp.h>
#include <ccutils/colors.h>

/**
 * ============================================================================
 * POPCORN KERNEL K-MEANS: OpenMP IMPLEMENTATION
 * ============================================================================
 * 
 * Paper: Popcorn: Accelerating Kernel K-means on GPUs through Sparse Linear Algebra
 * Conference: PPoPP '25
 * 
 * This implementation uses OpenMP for parallelization and properly implements
 * Equation 10 from the paper:
 * 
 *   D = -2·K·V^T + P̃ + C̃
 * 
 * ============================================================================
 * 
 * INTERFACE COMPATIBILITY:
 * This implementation has IDENTICAL function signatures to kernels_openblas.h
 * allowing runtime switching between OpenMP and OpenBLAS implementations.
 * 
 * ============================================================================
 */

/**
 * @brief Pair structure for (value, index) tuples
 */
struct Pair {
  float v;
  uint32_t i;
};

/**
 * @brief CPU argmin: returns pair with smaller value
 */
Pair cpu_argmin(Pair a, Pair b) {
  return a.v <= b.v ? a : b;
}

/**
 * @brief Finds the closest centroid for each point (OpenMP version)
 * 
 * For each point, finds the cluster with minimum distance.
 * Thread-safe cluster counting using atomic operations.
 *
 * Complexity: O(n·k)
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
    
    // Find minimum distance cluster
    for (uint32_t j = 0; j < k; j++) {
      float dist = distances[i * k + j];
      if (dist < min_dist) {
        min_dist = dist;
        min_cluster = j;
      }
    }
    
    points_clusters[i] = min_cluster;
    
    // Thread-safe increment
    #pragma omp atomic
    clusters_len[min_cluster]++;
  }
}

/**
 * @brief Compute kernel matrix K = P̂·P̂^T using OpenMP
 * 
 * Manual matrix multiplication with OpenMP collapse(2) for 2D parallelization.
 * Uses row-major storage for better cache locality.
 *
 * Complexity: O(n²·d)
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
  
  // K = P̂ * P̂^T (manual GEMM)
  #pragma omp parallel for collapse(2) default(none) \
          shared(K, points, n, d)
  for (uint64_t i = 0; i < n; i++) {
    for (uint64_t j = 0; j < n; j++) {
      float sum = 0.0f;
      for (uint32_t dim = 0; dim < d; dim++) {
        sum += points[i * d + dim] * points[j * d + dim];
      }
      K[i * n + j] = sum;
    }
  }
}

/**
 * @brief Apply linear kernel: K = -2·K element-wise
 * 
 * Multiplies all elements by -2.0 in parallel.
 * Simple element-wise operation.
 *
 * Complexity: O(n²)
 *
 * @param n matrix dimension (n×n)
 * @param K matrix to transform in-place
 */
void apply_linear_kernel_cpu(const uint32_t n, float* K) {
  #pragma omp parallel for default(none) shared(K, n)
  for (uint64_t i = 0; i < (uint64_t)n * n; i++) {
    K[i] *= -2.0f;
  }
}

/**
 * @brief Apply polynomial kernel: K = -2·(K + 1)²
 * 
 * Step 1: Add 1 to all elements
 * Step 2: Square and scale by -2
 *
 * Complexity: O(n²)
 *
 * @param n matrix dimension
 * @param K matrix to transform in-place
 */
void apply_polynomial_kernel_cpu(const uint32_t n, float* K) {
  // K = K + 1
  #pragma omp parallel for default(none) shared(K, n)
  for (uint64_t i = 0; i < (uint64_t)n * n; i++) {
    K[i] += 1.0f;
  }
  
  // K = -2 * K^2
  #pragma omp parallel for default(none) shared(K, n)
  for (uint64_t i = 0; i < (uint64_t)n * n; i++) {
    K[i] = -2.0f * K[i] * K[i];
  }
}

/**
 * @brief Apply sigmoid kernel: K = -2·tanh(K + 1)
 * 
 * Step 1: Add 1 to all elements
 * Step 2: Apply tanh and scale by -2
 *
 * Complexity: O(n²) with transcendental function
 *
 * @param n matrix dimension
 * @param K matrix to transform in-place
 */
void apply_sigmoid_kernel_cpu(const uint32_t n, float* K) {
  // K = K + 1
  #pragma omp parallel for default(none) shared(K, n)
  for (uint64_t i = 0; i < (uint64_t)n * n; i++) {
    K[i] += 1.0f;
  }
  
  // K = -2 * tanh(K)
  #pragma omp parallel for default(none) shared(K, n)
  for (uint64_t i = 0; i < (uint64_t)n * n; i++) {
    K[i] = -2.0f * std::tanh(K[i]);
  }
}

/**
 * @brief Extract diagonal of K: P̃[i] = K[i,i]
 * 
 * Copies diagonal elements into vector for point norms.
 * P̃_{i,j} = ||φ(p_i)||² (same value for all j).
 *
 * Complexity: O(n)
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
 * @brief Build CSR sparse matrix V from cluster assignments
 * 
 * V[j,i] = 1/|L_j| if point i is in cluster j, else 0
 * Stored in CSR format: (vals, colinds, rowptrs)
 *
 * Algorithm:
 * 1. Count non-zeros per row
 * 2. Compute cumulative row offsets
 * 3. Fill CSR arrays
 *
 * Complexity: O(n + k)
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
 * Implements Equation 10 from paper with four steps:
 * 
 * 1. SpMM: E = -2·K·V^T (sparse-dense multiplication)
 *    - For each sparse row j of V
 *    - Compute E[j,i] = -2·sum_ℓ(V[j,ℓ]·K[ℓ,i])
 *    
 * 2. Extract: z[i] = E[cluster[i], i]
 *    - Use cluster assignments to extract diagonal of E^T
 *    
 * 3. SpMV: c_norms = -0.5·V·z (sparse matrix-vector product)
 *    - For each sparse row j of V
 *    - Compute c_norms[j] = -0.5·sum_i(V[j,i]·z[i])
 *    
 * 4. Assemble: D[i,j] = E[j,i] + p_norms[i] + c_norms[j]
 *    - Transpose E and add outer products
 *
 * Complexity: O(n²) dominated by SpMM
 *
 * @param n number of points
 * @param k number of clusters
 * @param K kernel matrix (n×n, row-major)
 * @param p_norms point norms (n elements)
 * @param V_vals CSR values (n non-zeros)
 * @param V_colinds CSR column indices
 * @param V_rowptrs CSR row pointers (k+1 elements)
 * @param points_clusters cluster assignments (n elements)
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
  
  // ──────────────────────────────────────────────────────────────────────
  // STEP 1: Compute E = -2·K·V^T (SpMM)
  // ──────────────────────────────────────────────────────────────────────
  // V is k×n (sparse, CSR), K is n×n (dense, row-major)
  // E is k×n (dense, row-major)
  // For each sparse row j of V, compute E[j,:] = -2·V[j,:]·K
  
  std::vector<float> E(k * n, 0.0f);
  
  #pragma omp parallel for default(none) \
          shared(E, K, V_vals, V_colinds, V_rowptrs, n, k)
  for (uint32_t j = 0; j < k; j++) {
    // For cluster j (sparse row), compute E[j,i] for all i
    for (uint32_t i = 0; i < n; i++) {
      float sum = 0.0f;
      
      // V[j,:]·K[:,i] = sum over ℓ (V[j,ℓ]·K[ℓ,i])
      for (int32_t idx = V_rowptrs[j]; idx < V_rowptrs[j + 1]; idx++) {
        int32_t col = V_colinds[idx];
        sum += V_vals[idx] * K[col * n + i];
      }
      
      E[j * n + i] = -2.0f * sum;
    }
  }
  
  // ──────────────────────────────────────────────────────────────────────
  // STEP 2: Extract z[i] = E[cluster[i], i]
  // ──────────────────────────────────────────────────────────────────────
  // For each point i, extract the diagonal term for its cluster
  // z[i] = E[cluster[i], i] (used in SpMV for centroid norm computation)
  
  std::vector<float> z(n);
  #pragma omp parallel for default(none) \
          shared(z, E, points_clusters, n, k)
  for (uint32_t i = 0; i < n; i++) {
    uint32_t cluster = points_clusters[i];
    z[i] = E[cluster * n + i];
  }
  
  // ──────────────────────────────────────────────────────────────────────
  // STEP 3: Compute c_norms = -0.5·V·z (SpMV)
  // ──────────────────────────────────────────────────────────────────────
  // For each sparse row j of V, compute c_norms[j] = -0.5·sum_i(V[j,i]·z[i])
  // Leverages: V has exactly one non-zero per column
  // This is O(n) work since V has n non-zeros total
  
  std::vector<float> c_norms(k, 0.0f);
  #pragma omp parallel for default(none) \
          shared(c_norms, V_vals, V_colinds, V_rowptrs, z, k)
  for (uint32_t j = 0; j < k; j++) {
    float sum = 0.0f;
    
    // Compute V[j,:]·z = sum_i (V[j,i]·z[i])
    for (int32_t idx = V_rowptrs[j]; idx < V_rowptrs[j + 1]; idx++) {
      int32_t col = V_colinds[idx];
      sum += V_vals[idx] * z[col];
    }
    
    c_norms[j] = -0.5f * sum;
  }
  
  // ──────────────────────────────────────────────────────────────────────
  // STEP 4: Assemble D = E^T + P̃ + C̃
  // ──────────────────────────────────────────────────────────────────────
  // D[i,j] = E[j,i] + p_norms[i] + c_norms[j]
  // E is k×n, output D is n×k (transpose and add outer products)
  
  #pragma omp parallel for collapse(2) default(none) \
          shared(distances, E, p_norms, c_norms, n, k)
  for (uint32_t i = 0; i < n; i++) {
    for (uint32_t j = 0; j < k; j++) {
      distances[i * k + j] = E[j * n + i] + p_norms[i] + c_norms[j];
    }
  }
}

/**
 * @brief Alternative: Compute distances with explicit transpose (for comparison)
 * 
 * Same algorithm but with explicit matrix transpose step.
 * May be slower due to extra memory operations.
 */
void compute_distances_transpose_cpu(const uint32_t n,
                                    const uint32_t k,
                                    const float* K,
                                    const float* p_norms,
                                    const float* V_vals,
                                    const int32_t* V_colinds,
                                    const int32_t* V_rowptrs,
                                    const uint32_t* points_clusters,
                                    float* distances) {
  
  // Step 1-3: Same as above
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
  
  // Step 4: Explicit transpose
  std::vector<float> E_T(n * k);
  #pragma omp parallel for collapse(2) default(none) \
          shared(E_T, E, n, k)
  for (uint32_t i = 0; i < n; i++) {
    for (uint32_t j = 0; j < k; j++) {
      E_T[i * k + j] = E[j * n + i];
    }
  }
  
  // Step 5: Add p_norms and c_norms
  #pragma omp parallel for collapse(2) default(none) \
          shared(distances, E_T, p_norms, c_norms, n, k)
  for (uint32_t i = 0; i < n; i++) {
    for (uint32_t j = 0; j < k; j++) {
      distances[i * k + j] = E_T[i * k + j] + p_norms[i] + c_norms[j];
    }
  }
}

#endif // KERNELS_OPENMP_H