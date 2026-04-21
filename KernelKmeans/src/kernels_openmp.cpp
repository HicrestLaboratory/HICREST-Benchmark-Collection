#include <bits/stdc++.h>
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
    for (uint32_t j = 0; j < d; j++) {
      centroids[cluster * d + j] *= scale;
    }
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
 * @brief Compute kernel matrix K = P * P^T using naive multiplication
 * 
 * For each pair of points (i,j), computes their dot product.
 * Parallelized with collapse(2) to distribute both loops across threads.
 * 
 * @param K output: n x n kernel matrix
 * @param P input: n x d points matrix
 * @param n number of points
 * @param d dimensionality
 */
void compute_kernel_matrix_cpu(DATA_TYPE* K, 
                               const DATA_TYPE* P, 
                               const unsigned long long n, 
                               const uint32_t d) {
  
  #pragma omp parallel for collapse(2) default(none) \
          shared(K, P, n, d)
  for (unsigned long long i = 0; i < n; i++) {
    for (unsigned long long j = 0; j < n; j++) {
      DATA_TYPE result = 0.0;
      for (uint32_t dim = 0; dim < d; dim++) {
        result += P[i * d + dim] * P[j * d + dim];
      }
      K[i * n + j] = result;
    }
  }
}

/**
 * @brief Initialize kernel matrix using naive GEMM (no library)
 * 
 * Computes B = P * P^T manually without external libraries.
 * Memory layout is row-major for better cache locality.
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
  
  // B = P * P^T (manually optimized with blocking for cache)
  #pragma omp parallel for default(none) \
          shared(B, points, n, d)
  for (unsigned long long i = 0; i < n; i++) {
    for (unsigned long long j = 0; j < n; j++) {
      DATA_TYPE sum = 0.0;
      for (uint32_t k = 0; k < d; k++) {
        sum += points[i * d + k] * points[j * d + k];
      }
      B[i * n + j] = sum;
    }
  }
}

/**
 * @brief Apply linear kernel transformation: B *= -2.0
 * 
 * @param n matrix size (n x n)
 * @param d dimensionality (unused, kept for interface compatibility)
 * @param B matrix to transform in-place
 */
void apply_linear_kernel_cpu(const uint32_t n, const uint32_t d, DATA_TYPE* B) {
  (void)d; // Mark as intentionally unused
  
  #pragma omp parallel for default(none) \
          shared(B, n)
  for (unsigned long long i = 0; i < (unsigned long long)n * n; i++) {
    B[i] *= -2.0;
  }
}

/**
 * @brief Apply polynomial kernel transformation: B = -2 * (B + 1)^2
 * 
 * Implements a degree-2 polynomial kernel with specific coefficients
 * for kernel K-means. The parameter d is unused but kept for interface
 * compatibility.
 * 
 * @param n matrix size (n x n)
 * @param d dimensionality (unused, kept for interface compatibility)
 * @param B matrix to transform in-place
 */
void apply_polynomial_kernel_cpu(const uint32_t n, const uint32_t d, DATA_TYPE* B) {
  (void)d; // Mark as intentionally unused
  
  #pragma omp parallel for default(none) \
          shared(B, n)
  for (unsigned long long i = 0; i < (unsigned long long)n * n; i++) {
    DATA_TYPE x = B[i] + 1.0;
    B[i] = -2.0 * x * x;
  }
}

/**
 * @brief Apply sigmoid kernel transformation: B = -2 * tanh(B + 1)
 * 
 * Uses hyperbolic tangent as a smooth kernel activation function.
 * The parameter d is unused but kept for interface compatibility.
 * 
 * @param n matrix size (n x n)
 * @param d dimensionality (unused, kept for interface compatibility)
 * @param B matrix to transform in-place
 */
void apply_sigmoid_kernel_cpu(const uint32_t n, const uint32_t d, DATA_TYPE* B) {
  (void)d; // Mark as intentionally unused
  
  #pragma omp parallel for default(none) \
          shared(B, n)
  for (unsigned long long i = 0; i < (unsigned long long)n * n; i++) {
    B[i] = -2.0 * std::tanh(B[i] + 1.0);
  }
}

/**
 * @brief Extract diagonal elements with scaling
 * 
 * Copies diagonal elements from M to output, scaling by 1/alpha.
 * Useful for extracting and normalizing self-similarity values.
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
 * @brief Compute distances using sparse-dense matrix multiplication
 * 
 * Computes D = V * B where V is stored in CSR format.
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
    for (uint32_t j = 0; j < n; j++) {
      DATA_TYPE sum = 0.0;
      
      // Iterate over non-zero elements in row i of V
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
  
  #pragma omp parallel for default(none) \
          shared(K, clusters, clusters_len, distances, n, k)
  for (uint32_t point_id = 0; point_id < n; point_id++) {
    for (uint32_t cluster = 0; cluster < k; cluster++) {
      DATA_TYPE sum = 0.0;
      
      // Sum K[point_id][j] for all j in this cluster
      for (uint32_t j = 0; j < n; j++) {
        if (clusters[j] == static_cast<int32_t>(cluster)) {
          sum += K[point_id * n + j];
        }
      }
      
      // Average by cluster size
      distances[point_id * k + cluster] = sum / static_cast<DATA_TYPE>(clusters_len[cluster]);
    }
  }
}

/**
 * @brief Sum transformed kernel values for centroids in naive method
 * 
 * Computes the diagonal terms needed for distance calculation in the
 * naive kernel K-means approach.
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
 * @brief Compute final distances using naive method
 * 
 * Combines point sums and centroid contributions to get final distances.
 * Formula: distance[i,j] = tmp[i,j] + centroids[j]
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
  
  #pragma omp parallel for collapse(2) default(none) \
          shared(K, centroids, tmp, distances, n, k)
  for (uint32_t i = 0; i < n; i++) {
    for (uint32_t j = 0; j < k; j++) {
      distances[j + i * k] = tmp[j + i * k] + centroids[j];
    }
  }
}