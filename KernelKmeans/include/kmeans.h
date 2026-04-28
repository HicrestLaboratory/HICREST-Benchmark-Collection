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

/**
 * ============================================================================
 * KERNEL K-MEANS CPU IMPLEMENTATION
 * ============================================================================
 * 
 * Implements Popcorn: Accelerating Kernel K-means on GPUs through Sparse 
 * Linear Algebra (PPoPP '25) using either OpenBLAS or OpenMP backends.
 * 
 * Key Algorithm (Equation 10):
 *   D = -2·K·V^T + P̃ + C̃
 * 
 * Where:
 *   K = kernel matrix (n×n)
 *   V = sparse cluster membership (k×n, CSR)
 *   P̃ = point norms ||φ(p_i)||²
 *   C̃ = centroid norms ||c_j||²
 *   D = distance matrix (n×k, row-major)
 * ============================================================================
 */

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
     * Uses the optimized Popcorn linear algebra formulation for distance computation.
     * 
     * Both OpenBLAS and OpenMP backends are supported - choose via compile-time flag.
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
           Point<float>** _points,
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
     * Both OpenMP and OpenBLAS implementations use the same Popcorn algorithm
     * with optimized linear algebra formulation for distance computation.
     * 
     * Main Loop:
     * 1. Compute distances: D = -2·K·V^T + P̃ + C̃ (Popcorn formula)
     * 2. Assign points to nearest clusters
     * 3. Update cluster membership matrix V
     * 4. Compute objective score
     * 5. Check convergence
     * 
     * @param maxiter maximum number of iterations
     * @param check_converged whether to check for convergence (early stopping)
     * @return number of iterations until convergence (or maxiter if not converged)
     */
    uint64_t run(uint64_t maxiter, bool check_converged);

    /**
     * @brief Get the final objective score
     * 
     * @return sum of distances from points to their assigned clusters
     */
    double get_score() const { return score; }
    
    /**
     * @brief Get cluster assignments for all points
     * 
     * @return vector of cluster IDs (0 to k-1) for each point
     */
    const std::vector<uint32_t>& get_points_clusters() const { 
        return clusters; 
    }

private:
    // ────────────────────────────────────────────────────────────────
    // DATA DIMENSIONS & PARAMETERS
    // ────────────────────────────────────────────────────────────────
    
    const size_t n;           // Number of points
    const uint32_t d;         // Number of dimensions (features)
    const uint32_t k;         // Number of clusters
    const float tol;          // Convergence tolerance
    const InitMethod initMethod;
    const Kernel kernel;

    // ────────────────────────────────────────────────────────────────
    // MAIN DATA STRUCTURES
    // ────────────────────────────────────────────────────────────────
    
    // Point data (n×d, row-major)
    std::vector<float> points;
    
    // Kernel matrix (n×n, row-major)
    // K[i,j] = κ(p_i, p_j) after kernel transformation
    std::vector<float> K;
    
    // Point norms: P̃[i] = ||φ(p_i)||² (diagonal of K)
    std::vector<float> p_norms;
    
    // Cluster assignments: clusters[i] = cluster ID for point i
    std::vector<uint32_t> clusters;
    
    // Cluster sizes: clusters_len[j] = number of points in cluster j
    std::vector<uint32_t> clusters_len;
    
    // Sparse matrix V in CSR format (k×n, exactly n non-zeros)
    // V[j,i] = 1/|L_j| if point i is in cluster j, else 0
    std::vector<float> V_vals;              // CSR values
    std::vector<int32_t> V_colinds;         // CSR column indices
    std::vector<int32_t> V_rowptrs;         // CSR row pointers
    
    // Distance matrix (n×k, row-major)
    // distances[i*k + j] = distance from point i to cluster j
    std::vector<float> distances;

    // ────────────────────────────────────────────────────────────────
    // ALGORITHM STATE
    // ────────────────────────────────────────────────────────────────
    
    double score;              // Current objective value
    double last_score;         // Previous iteration's objective
    
    std::mt19937* generator;   // Random number generator

    // ────────────────────────────────────────────────────────────────
    // PRIVATE METHODS
    // ────────────────────────────────────────────────────────────────

    /**
     * @brief Initialize kernel matrix and preprocessing
     * 
     * Computes:
     * 1. K = P̂·P̂^T using GEMM
     * 2. Apply kernel transformation (linear, poly, or sigmoid)
     * 3. Extract point norms from diagonal
     * 
     * This is done once before the main loop since K and p_norms
     * are constant throughout iterations.
     */
    void initialize_kernel_matrix();

    /**
     * @brief Initialize cluster assignments
     * 
     * Uses specified initialization method:
     * - Random: round-robin assignment
     * - K-means++: probabilistic initialization (falls back to random on CPU)
     */
    void initialize_clusters();

    /**
     * @brief Compute distances using Popcorn formula
     * 
     * Implements complete Equation 10:
     *   D = -2·K·V^T + P̃ + C̃
     * 
     * Uses sparse-dense matrix multiplication for efficiency.
     * This is the most computationally expensive step each iteration.
     */
    void compute_distances();

    /**
     * @brief Build sparse matrix V (cluster membership)
     * 
     * Constructs V in CSR format from current cluster assignments.
     * V[j,i] = 1/|L_j| if point i is in cluster j
     * 
     * This is updated after cluster assignments change since V encodes
     * the current clustering.
     */
    void compute_v_matrix();
};

#endif // __KMEANS_CPU_H__