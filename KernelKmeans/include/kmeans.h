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

    // Constructor
    Kmeans(const size_t _n, const uint32_t _d, const uint32_t _k,
           const float _tol, const int* seed,
           Point<DATA_TYPE>** _points,
           const InitMethod _initMethod,
           const Kernel _kernel,
           const int _level);

    // Destructor
    ~Kmeans();

    // Main clustering function
    uint64_t run(uint64_t maxiter, bool check_converged);

    // Getters
    double get_score() const { return score; }
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
    DATA_TYPE* B;

    // Clustering results
    std::vector<uint32_t> h_points_clusters;  // Cluster assignment for each point
    int32_t* clusters;        // Cluster assignments (n)
    uint32_t* clusters_len;   // Number of points per cluster (k)

    // Sparse matrix V (CSR format) - k x n
    DATA_TYPE* V_vals;
    int32_t* V_colinds;
    int32_t* V_rowptrs;

    // Distance matrix and norms
    DATA_TYPE* distances;     // Distance matrix (k x n in column major)
    DATA_TYPE* points_row_norms;  // Row norms of points
    DATA_TYPE* centroids_row_norms;  // Row norms of centroids

    // Temporary buffers
    DATA_TYPE* z_vals;        // Temporary vector for SpMV

    // Algorithm parameters
    const InitMethod initMethod;
    const Kernel kernel;
    const int level;

    // Convergence tracking
    double score;
    double last_score;

    // Random number generator
    std::mt19937* generator;

    // Initialization methods
    void init_centroids_rand();
    void init_centroids_plus_plus();

    // Kernel initialization
    template <typename KernelType>
    void init_kernel_matrix();

    // Distance computation methods
    void compute_distances_naive();
    void compute_distances_optimized();

    // Helper functions
    void compute_v_matrix();
    void compute_points_norms();
    void compute_centroids_norms();
};

#endif // __KMEANS_CPU_H__