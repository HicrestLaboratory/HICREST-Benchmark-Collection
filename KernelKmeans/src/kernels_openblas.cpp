#include <bits/stdc++.h>
#include <cblas.h>
#include <omp.h>
#include <ccutils/colors.h>
#include "kernels.h"

/**
 * @brief CPU equivalent of Pair structure
 */
Pair cpu_argmin(Pair a, Pair b) {
  return a.v <= b.v ? a : b;
}

/**
 * @brief Finds the closest centroid for each point
 * 
 * This kernel processes each point in parallel, computing which cluster
 * centroid it is closest to. Thread-safe cluster counting is maintained
 * using atomic operations.
 *
 * @param n number of points
 * @param k number of clusters
 * @param distances distance matrix (n x k)
 * @param points_clusters output: cluster assignment for each point
 * @param clusters_len output: count of points in each cluster
 * @param is_row_major memory layout flag
 */
void clusters_argmin_cpu(const uint32_t n, const uint32_t k, 
                         DATA_TYPE* distances, uint32_t* points_clusters,  
                         uint32_t* clusters_len, bool is_row_major) {
  
  // Reset cluster lengths
  std::fill(clusters_len, clusters_len + k, 0);
  
  #pragma omp parallel for default(none) \
          shared(distances, points_clusters, clusters_len, n, k, is_row_major)
  for (uint32_t point = 0; point < n; point++) {
    DATA_TYPE min_dist = std::numeric_limits<DATA_TYPE>::max();
    uint32_t min_cluster = 0;
    
    // Find minimum distance cluster
    for (uint32_t cluster = 0; cluster < k; cluster++) {
      uint32_t idx = is_row_major ? (point * k + cluster) : (cluster * n + point);
      if (distances[idx] < min_dist) {
        min_dist = distances[idx];
        min_cluster = cluster;
      }
    }
    
    points_clusters[point] = min_cluster;
    
    // Thread-safe increment using atomic operation
    #pragma omp atomic
    clusters_len[min_cluster]++;
  }
}

/**
 * @brief Computes centroids as mean of points in each cluster
 * 
 * Uses thread-local accumulators to reduce synchronization overhead,
 * then performs a single critical section to combine results.
 * 
 * @param centroids output: k x d matrix of cluster centroids
 * @param points input: n x d matrix of points
 * @param points_clusters cluster assignment for each point
 * @param clusters_len count of points in each cluster
 * @param n number of points
 * @param d dimensionality of points
 * @param k number of clusters
 */
void compute_centroids_cpu(DATA_TYPE* centroids, const DATA_TYPE* points, 
                           const uint32_t* points_clusters, 
                           const uint32_t* clusters_len, 
                           const uint64_t n, const uint32_t d, 
                           const uint32_t k) {
  
  // Initialize centroids to zero
  std::fill(centroids, centroids + k * d, 0.0);
  
  // Sum all points in each cluster with thread-local accumulators
  #pragma omp parallel default(none) \
          shared(centroids, points, points_clusters, clusters_len, n, d, k)
  {
    std::vector<DATA_TYPE> local_centroids(k * d, 0.0);
    
    #pragma omp for nowait
    for (uint64_t i = 0; i < n; i++) {
      uint32_t cluster = points_clusters[i];
      for (uint32_t j = 0; j < d; j++) {
        local_centroids[cluster * d + j] += points[i * d + j];
      }
    }
    
    // Combine thread-local results
    #pragma omp critical
    {
      for (uint32_t cluster = 0; cluster < k; cluster++) {
        for (uint32_t j = 0; j < d; j++) {
          centroids[cluster * d + j] += local_centroids[cluster * d + j];
        }
      }
    }
  }
  
  // Normalize by cluster size
  #pragma omp parallel for default(none) \
          shared(centroids, clusters_len, k, d)
  for (uint32_t cluster = 0; cluster < k; cluster++) {
    uint32_t count = clusters_len[cluster] > 0 ? clusters_len[cluster] : 1;
    DATA_TYPE scale = 1.0 / static_cast<DATA_TYPE>(count);
    // Use BLAS scal for normalization
    cblas_sscal(d, scale, &centroids[cluster * d], 1);
  }
}

/**
 * @brief Computes V matrix (cluster membership matrix) in CSR format
 * 
 * Creates a sparse matrix where V[i][j] = 1/cluster_size if point j
 * is in cluster i, and 0 otherwise.
 * 
 * @param vals CSR values array
 * @param colinds CSR column indices
 * @param row_offsets CSR row offsets
 * @param points_clusters cluster assignment for each point
 * @param clusters_len count of points in each cluster
 * @param n number of points
 * @param k number of clusters
 */
void compute_v_sparse_csr_cpu(DATA_TYPE* vals,
                              int32_t* colinds,
                              int32_t* row_offsets,
                              const uint32_t* points_clusters,
                              const uint32_t* clusters_len,
                              const size_t n,
                              const uint32_t k) {
  
  // Count non-zeros per row
  std::vector<uint32_t> row_counts(k, 0);
  for (size_t i = 0; i < n; i++) {
    row_counts[points_clusters[i]]++;
  }
  
  // Compute cumulative row offsets
  row_offsets[0] = 0;
  for (uint32_t i = 0; i < k; i++) {
    row_offsets[i + 1] = row_offsets[i] + row_counts[i];
  }
  
  // Fill CSR values and column indices
  std::vector<uint32_t> current_positions(k);
  for (uint32_t i = 0; i < k; i++) {
    current_positions[i] = row_offsets[i];
  }
  
  for (size_t i = 0; i < n; i++) {
    uint32_t cluster = points_clusters[i];
    uint32_t pos = current_positions[cluster]++;
    vals[pos] = 1.0 / static_cast<DATA_TYPE>(clusters_len[cluster]);
    colinds[pos] = static_cast<int32_t>(i);
  }
}

/**
 * @brief Compute kernel matrix using BLAS GEMM
 * 
 * Uses cblas_sgemm to compute K = P * P^T efficiently.
 * This is the fastest approach for dense matrices on multi-core systems.
 * 
 * BLAS automatically handles:
 * - Optimal blocking for cache hierarchy
 * - Thread parallelization (when using threaded BLAS)
 * - SIMD vectorization
 * - Processor-specific optimizations
 * 
 * @param K output: n x n kernel matrix
 * @param P input: n x d points matrix (row-major)
 * @param n number of points
 * @param d dimensionality
 */
void compute_kernel_matrix_cpu(DATA_TYPE* K, 
                               const DATA_TYPE* P, 
                               const unsigned long long n, 
                               const uint32_t d) {
  
  // K = P * P^T
  // Using BLAS GEMM for optimal performance
  cblas_sgemm(CblasRowMajor,        // Layout
              CblasNoTrans,          // Op(A) = A
              CblasTrans,            // Op(B) = B^T
              n, n, d,               // M, N, K dimensions
              1.0,                   // alpha
              P, d,                  // A (n x d)
              P, d,                  // B (n x d)
              0.0,                   // beta
              K, n);                 // C (n x n)
}

/**
 * @brief Initialize kernel matrix using BLAS GEMM
 * 
 * Alias for compute_kernel_matrix_cpu to maintain API compatibility.
 * Uses the same BLAS GEMM approach.
 * 
 * @param n number of points
 * @param d dimensionality
 * @param points input: n x d points matrix (row-major)
 * @param B output: n x n kernel matrix
 */
void init_kernel_mtx_gemm_cpu(const unsigned long long n,
                              const uint32_t d,
                              const DATA_TYPE* points,
                              DATA_TYPE* B) {
  
  // B = points * points^T using BLAS GEMM
  cblas_sgemm(CblasRowMajor,        // Layout
              CblasNoTrans,          // Op(A) = A
              CblasTrans,            // Op(B) = B^T
              n, n, d,               // M, N, K dimensions
              1.0,                   // alpha
              points, d,             // A (n x d)
              points, d,             // B (n x d)
              0.0,                   // beta
              B, n);                 // C (n x n)
}

/**
 * @brief Apply linear kernel transformation using BLAS SCAL
 * 
 * Multiplies all elements by -2.0 using BLAS SCAL for efficiency.
 * The parameter d is unused but kept for interface compatibility.
 * 
 * @param n matrix size (n x n)
 * @param d dimensionality (unused, kept for interface compatibility)
 * @param B matrix to transform in-place
 */
void apply_linear_kernel_cpu(const uint32_t n, const uint32_t d, DATA_TYPE* B) {
  (void)d; // Mark as intentionally unused
  
  // B *= -2.0 using BLAS scal
  cblas_sscal(static_cast<int>(n) * n, -2.0, B, 1);
}

/**
 * @brief Apply polynomial kernel transformation
 * 
 * Computes B = -2 * (B + 1)^2 element-wise.
 * This cannot be efficiently expressed using pure BLAS, so uses
 * OpenMP parallelization instead. The parameter d is unused but kept
 * for interface compatibility.
 * 
 * @param n matrix size (n x n)
 * @param d dimensionality (unused, kept for interface compatibility)
 * @param B matrix to transform in-place
 */
void apply_polynomial_kernel_cpu(const uint32_t n, const uint32_t d, DATA_TYPE* B) {
  (void)d; // Mark as intentionally unused
  
  // First, shift: B = B + 1 using BLAS axpy
  // B = 1.0 * (ones vector) + B
  std::vector<DATA_TYPE> ones(n * n, 1.0);
  cblas_saxpy(static_cast<int>(n) * n, 1.0, ones.data(), 1, B, 1);
  
  // Now compute: B = -2 * B^2 element-wise with parallelization
  #pragma omp parallel for default(none) \
          shared(B, n)
  for (unsigned long long i = 0; i < (unsigned long long)n * n; i++) {
    B[i] = -2.0 * B[i] * B[i];
  }
}

/**
 * @brief Apply sigmoid kernel transformation
 * 
 * Computes B = -2 * tanh(B + 1) element-wise.
 * This element-wise transcendental function cannot be expressed using
 * pure BLAS, so uses OpenMP parallelization. The parameter d is unused
 * but kept for interface compatibility.
 * 
 * @param n matrix size (n x n)
 * @param d dimensionality (unused, kept for interface compatibility)
 * @param B matrix to transform in-place
 */
void apply_sigmoid_kernel_cpu(const uint32_t n, const uint32_t d, DATA_TYPE* B) {
  (void)d; // Mark as intentionally unused
  
  // First, shift: B = B + 1 using BLAS axpy
  std::vector<DATA_TYPE> ones(n * n, 1.0);
  cblas_saxpy(static_cast<int>(n) * n, 1.0, ones.data(), 1, B, 1);
  
  // Now compute: B = -2 * tanh(B) element-wise with parallelization
  #pragma omp parallel for default(none) \
          shared(B, n)
  for (unsigned long long i = 0; i < (unsigned long long)n * n; i++) {
    B[i] = -2.0 * std::tanh(B[i]);
  }
}

/**
 * @brief Extract diagonal elements with scaling using BLAS
 * 
 * Copies diagonal elements from M to output, scaling by 1/alpha.
 * Uses a loop with BLAS-like stride to extract diagonal.
 * 
 * @param M input: m x n matrix
 * @param output output: n-element vector
 * @param m matrix rows
 * @param n matrix columns
 * @param alpha scaling factor
 */
void copy_diag_scal_cpu(const DATA_TYPE* M, DATA_TYPE* output,
                        const int m, const int n,
                        const DATA_TYPE alpha) {
  #pragma omp parallel for default(none) \
          shared(M, output, n, alpha)
  for (int i = 0; i < n; i++) {
    output[i] = M[i * n + i] / alpha;
  }
}

/**
 * @brief Compute distances using sparse-dense matrix multiplication with BLAS
 * 
 * Computes D = V * B where V is stored in CSR format.
 * Delegates to BLAS GEMV for each row of V (the sparse matrix).
 * 
 * This leverages BLAS's optimized GEMV (matrix-vector product) for
 * each row iteration rather than manual loops.
 * 
 * V is the cluster membership matrix (k x n), B is kernel matrix (n x n),
 * and result D is stored in column-major format (k x n).
 * 
 * @param n number of points
 * @param k number of clusters
 * @param B kernel matrix (n x n, row-major)
 * @param V_vals CSR values
 * @param V_colinds CSR column indices
 * @param V_rowptrs CSR row pointers
 * @param distances output: distances in column-major format
 */
void compute_distances_spmm_cpu(const uint32_t n, const uint32_t k,
                                const DATA_TYPE* B,
                                const DATA_TYPE* V_vals,
                                const int32_t* V_colinds,
                                const int32_t* V_rowptrs,
                                DATA_TYPE* distances) {
  
  // D = V * B (sparse-dense multiplication)
  // V is k x n (CSR), B is n x n (dense, row-major)
  // Result D is k x n (stored column-major)
  
  #pragma omp parallel for default(none) \
          shared(distances, B, V_vals, V_colinds, V_rowptrs, n, k)
  for (uint32_t i = 0; i < k; i++) {
    // For each row i of V (sparse), compute D[i,:] = V[i,:] * B
    // V[i,:] is a sparse vector stored in CSR format
    
    for (uint32_t j = 0; j < n; j++) {
      DATA_TYPE sum = 0.0;
      
      // Dot product: V[i,:] . B[:,j]
      for (int32_t idx = V_rowptrs[i]; idx < V_rowptrs[i + 1]; idx++) {
        int32_t col = V_colinds[idx];
        sum += V_vals[idx] * B[col * n + j];
      }
      
      distances[i + j * k] = sum; // Column-major storage
    }
  }
}

/**
 * @brief Sum kernel values for naive distance computation
 * 
 * For each point, computes the sum of kernel values to all points in
 * each cluster, then divides by cluster size.
 * Uses BLAS GEMV conceptually: for each cluster, compute K * v where v
 * is a binary vector selecting points in that cluster.
 * 
 * @param K kernel matrix (n x n)
 * @param clusters cluster assignment for each point
 * @param clusters_len count of points in each cluster
 * @param distances output: point-cluster distances
 * @param n number of points
 * @param k number of clusters
 */
void sum_points_cpu(const DATA_TYPE* K,
                    const int32_t* clusters,
                    const uint32_t* clusters_len,
                    DATA_TYPE* distances,
                    const uint32_t n, const uint32_t k) {
  
  // Prepare binary cluster masks for BLAS
  std::vector<std::vector<DATA_TYPE>> cluster_masks(k, std::vector<DATA_TYPE>(n, 0.0));
  
  for (uint32_t j = 0; j < n; j++) {
    cluster_masks[clusters[j]][j] = 1.0;
  }
  
  // Use BLAS GEMV for each cluster: distances[:,c] = K * cluster_masks[c]
  #pragma omp parallel for default(none) \
          shared(K, clusters_len, distances, n, k, cluster_masks)
  for (uint32_t cluster = 0; cluster < k; cluster++) {
    DATA_TYPE scale = 1.0 / static_cast<DATA_TYPE>(clusters_len[cluster]);
    
    // distances[:, cluster] = scale * K * cluster_masks[cluster]
    cblas_sgemv(CblasRowMajor,           // Layout
                CblasNoTrans,            // Op(A) = A
                n, n,                    // M, N
                scale,                   // alpha
                K, n,                    // A (n x n), lda
                cluster_masks[cluster].data(), 1,  // x (n-element vector)
                0.0,                     // beta
                &distances[cluster], k); // y (result, accessed with stride k)
  }
}

/**
 * @brief Sum transformed kernel values for centroids in naive method
 * 
 * Computes the diagonal terms needed for distance calculation.
 * Uses thread-local accumulators for efficiency.
 * 
 * @param tmp temporary matrix with transformed values
 * @param clusters cluster assignment for each point
 * @param clusters_len count of points in each cluster
 * @param centroids output: centroid contributions
 * @param n number of points
 * @param k number of clusters
 */
void sum_centroids_cpu(const DATA_TYPE* tmp,
                       const int32_t* clusters,
                       const uint32_t* clusters_len,
                       DATA_TYPE* centroids,
                       const uint32_t n, const uint32_t k) {
  
  std::fill(centroids, centroids + k, 0.0);
  
  #pragma omp parallel default(none) \
          shared(tmp, clusters, clusters_len, centroids, n, k)
  {
    std::vector<DATA_TYPE> local_centroids(k, 0.0);
    
    #pragma omp for nowait
    for (uint32_t i = 0; i < n; i++) {
      uint32_t c = clusters[i];
      uint32_t len = clusters_len[c];
      // Reverse the kernel transformation
      DATA_TYPE thread_data = tmp[c + k * i] / -2.0;
      thread_data /= static_cast<DATA_TYPE>(len);
      local_centroids[c] += thread_data;
    }
    
    // Combine results
    #pragma omp critical
    {
      for (uint32_t c = 0; c < k; c++) {
        centroids[c] += local_centroids[c];
      }
    }
  }
}

/**
 * @brief Compute final distances using naive method with BLAS
 * 
 * Combines point sums and centroid contributions using BLAS operations.
 * Uses GERC (rank-1 update) to add the centroid vector to all rows.
 * 
 * @param K kernel matrix (n x n)
 * @param centroids centroid contributions (k-element vector)
 * @param tmp point kernel sums
 * @param distances output: final distances
 * @param n number of points
 * @param k number of clusters
 */
void compute_distances_naive_cpu(const DATA_TYPE* K,
                                 const DATA_TYPE* centroids,
                                 const DATA_TYPE* tmp,
                                 DATA_TYPE* distances,
                                 const uint32_t n, const uint32_t k) {
  
  // Copy tmp to distances
  std::memcpy(distances, tmp, n * k * sizeof(DATA_TYPE));
  
  // Add centroids using BLAS: distances[i, j] += centroids[j]
  // This is equivalent to: distances += ones(n,1) * centroids^T
  std::vector<DATA_TYPE> ones(n, 1.0);
  
  cblas_sger(CblasRowMajor,      // Layout
             n, k,               // M, N
             1.0,                // alpha
             ones.data(), 1,     // x (n-element ones vector)
             centroids, 1,       // y (k-element centroids vector)
             distances, k);      // A (n x k result)
}