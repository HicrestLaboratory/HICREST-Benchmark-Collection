#ifndef __KMEANS_CPU_H__
#define __KMEANS_CPU_H__

#include <vector>
#include <random>
#include <fstream>
#include <algorithm>
#include <numeric>

#include "common.h"
#include "point.hpp"
#include "kernels.h"

class Kmeans {
public:
    enum class InitMethod {
        random,
        plus_plus
    };

    enum class Kernel {
        linear,
        polynomial,
        sigmoid
    };

    /**
     * @brief Constructor for kernel K-means clustering
     * 
     * Initializes the clustering algorithm with the given points and parameters.
     * Uses the optimized linear algebra formulation for distance computation.
     * 
     * @param _n number of points
     * @param _d dimensionality of points
     * @param _k number of clusters
     * @param _tol convergence tolerance
     * @param seed random seed (nullptr for auto-seeding)
     * @param _points array of point objects
     * @param _initMethod centroid initialization method (random or k-means++)
     * @param _kernel kernel type (linear, polynomial, or sigmoid)
     */
    Kmeans(const size_t _n, const uint32_t _d, const uint32_t _k,
           const float _tol, const int* seed,
           Point<DATA_TYPE>** _points,
           const InitMethod _initMethod,
           const Kernel _kernel);

    /**
     * @brief Destructor
     */
    ~Kmeans();

    /**
     * @brief Main clustering loop
     * 
     * Executes the kernel K-means algorithm for up to maxiter iterations.
     * Both OpenMP and OpenBLAS implementations use the same optimized
     * linear algebra formulation for distance computation.
     * 
     * @param maxiter maximum number of iterations
     * @param check_converged whether to check for convergence
     * @return number of iterations until convergence (or maxiter if not converged)
     */
    uint64_t run(uint64_t maxiter, bool check_converged);

    // Getters
    /**
     * @brief Get the final objective score
     * @return sum of distances from points to their assigned clusters
     */
    double get_score() const { return score; }
    
    /**
     * @brief Get cluster assignments for all points
     * @return vector of cluster IDs (0 to k-1) for each point
     */
    const std::vector<uint32_t>& get_points_clusters() const { return h_points_clusters; }

private:
    // Data dimensions
    const size_t n;           // Number of points
    const uint32_t d;         // Number of dimensions
    const uint32_t k;         // Number of clusters
    const float tol;          // Convergence tolerance

    // Memory sizes
    const size_t POINTS_BYTES;
    const size_t CENTROIDS_BYTES;

    // Point data
    DATA_TYPE* h_points;      // Host points (n x d)
    Point<DATA_TYPE>** points; // Original point objects

    // Kernel matrix (n x n)
    // B[i*n + j] = K(x_i, x_j) where K is the kernel function
    DATA_TYPE* B;

    // Clustering results
    std::vector<uint32_t> h_points_clusters;  // Cluster assignment for each point
    int32_t* clusters;        // Cluster assignments (n)
    uint32_t* clusters_len;   // Number of points per cluster (k)

    // Sparse matrix V (CSR format) - k x n
    // V is the cluster membership matrix where V[i,j] = 1/|C_i| if point j in cluster i
    DATA_TYPE* V_vals;        // CSR values
    int32_t* V_colinds;       // CSR column indices
    int32_t* V_rowptrs;       // CSR row pointers

    // Distance matrix and norms
    DATA_TYPE* distances;     // Distance matrix (k x n in column-major format)
    DATA_TYPE* points_row_norms;     // Row norms of points: p_tilde[i]
    DATA_TYPE* centroids_row_norms;  // Row norms of centroids: c_tilde[j]

    // Temporary buffers
    DATA_TYPE* z_vals;        // Temporary vector for SpMV: z[i] = D[cluster[i], i]

    // Algorithm parameters
    const InitMethod initMethod;
    const Kernel kernel;

    // Convergence tracking
    double score;
    double last_score;

    // Random number generator
    std::mt19937* generator;

    /**
     * @brief Initialize centroids using random assignment
     * 
     * Assigns each point to a cluster in round-robin fashion.
     * Then builds the sparse V matrix for distance computation.
     */
    void init_centroids_rand();

    /**
     * @brief Initialize centroids using K-means++ algorithm
     * 
     * Currently falls back to random initialization on CPU.
     */
    void init_centroids_plus_plus();

    /**
     * @brief Template function to initialize kernel matrix
     * 
     * Computes B = K(x_i, x_j) using GEMM-based approach
     * and applies the kernel transformation.
     * 
     * @tparam KernelType kernel function type (LinearKernel, PolynomialKernel, SigmoidKernel)
     */
    template <typename KernelType>
    void init_kernel_matrix();

    /**
     * @brief Compute distances using optimized linear algebra formulation
     * 
     * Implements kernel K-means distance computation:
     * D[i,j] = ||phi(x_i) - mu_j||^2
     *        = p_tilde[i] + d_tilde[i,j] + c_tilde[j]
     * 
     * Steps:
     * 1. D_temp = V * B (sparse-dense matrix multiplication)
     * 2. z[i] = D_temp[cluster[i], i] (extract assigned cluster distances)
     * 3. c_tilde = -0.5 * V * z (compute centroid norms)
     * 4. D[i,j] += c_tilde[j] (add centroid norms)
     * 
     * Both OpenMP and OpenBLAS implementations use this same algorithm
     * but with different primitive implementations.
     */
    void compute_distances();

    /**
     * @brief Build sparse matrix V in CSR format
     * 
     * Creates the cluster membership matrix from current cluster assignments.
     * V[i,j] = 1/|C_i| if point j is in cluster i, 0 otherwise.
     */
    void compute_v_matrix();

    /**
     * @brief Compute point row norms
     * 
     * Extracts diagonal elements from kernel matrix:
     * p_tilde[i] = B[i,i] / (-2.0)
     */
    void compute_points_norms();

    /**
     * @brief Compute centroid row norms
     * 
     * Computes c_tilde = -0.5 * V * z using sparse matrix-vector product,
     * where z[i] = D[cluster[i], i].
     */
    void compute_centroids_norms();
};

#endif // __KMEANS_CPU_H__