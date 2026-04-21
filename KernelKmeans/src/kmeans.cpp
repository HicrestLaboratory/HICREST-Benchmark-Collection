#include "kmeans.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <omp.h>

Kmeans::Kmeans(const size_t _n, const uint32_t _d, const uint32_t _k,
               const float _tol, const int* seed,
               Point<DATA_TYPE>** _points,
               const InitMethod _initMethod,
               const Kernel _kernel,
               const int _level)
    : n(_n), d(_d), k(_k), tol(_tol),
      POINTS_BYTES(_n * _d * sizeof(DATA_TYPE)),
      CENTROIDS_BYTES(_k * _d * sizeof(DATA_TYPE)),
      h_points_clusters(_n),
      points(_points),
      initMethod(_initMethod),
      kernel(_kernel),
      level(_level),
      score(0.0),
      last_score(0.0) {

    // Initialize random generator
    if (seed != nullptr) {
        generator = new std::mt19937(*seed);
    } else {
        std::random_device rd;
        generator = new std::mt19937(rd());
    }

    // Allocate and copy points
    h_points = new DATA_TYPE[n * d];
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d; ++j) {
            h_points[i * d + j] = _points[i]->get(j);
        }
    }

#if LOG
    std::ofstream points_out;
    points_out.open("points-cpu.out");
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < d; j++) {
            points_out << h_points[i * d + j] << ",";
        }
        points_out << std::endl;
    }
    points_out.close();
#endif

    // Allocate kernel matrix
    B = new DATA_TYPE[n * n];
    std::fill(B, B + n * n, 0.0);

    // Allocate cluster tracking
    clusters = new int32_t[n];
    clusters_len = new uint32_t[k];
    std::fill(clusters_len, clusters_len + k, 0);

    // Initialize kernel matrix based on kernel type
    std::cout << "Initializing kernel matrix..." << std::endl;
    switch (kernel) {
        case Kernel::linear:
            init_kernel_matrix<LinearKernel>();
            break;
        case Kernel::polynomial:
            init_kernel_matrix<PolynomialKernel>();
            break;
        case Kernel::sigmoid:
            init_kernel_matrix<SigmoidKernel>();
            break;
    }

#ifdef LOG_KERNEL
    std::ofstream kernel_out;
    kernel_out.open("kernel-cpu.out");
    for (size_t i = 0; i < std::min(n, (size_t)10); i++) {
        for (size_t j = 0; j < n; j++) {
            kernel_out << B[i * n + j] << ",";
        }
        kernel_out << std::endl;
    }
    kernel_out.close();
#endif

    // Allocate sparse matrix V (CSR format)
    V_vals = new DATA_TYPE[n];
    V_colinds = new int32_t[n];
    V_rowptrs = new int32_t[k + 1];

    // Allocate distance matrix and norms
    distances = new DATA_TYPE[k * n];  // Column major: k rows, n columns
    points_row_norms = new DATA_TYPE[n];
    centroids_row_norms = new DATA_TYPE[k];
    z_vals = new DATA_TYPE[n];

    // Compute point norms from diagonal of B
    compute_points_norms();

    // Initialize centroids
    std::cout << "Initializing centroids..." << std::endl;
    switch (initMethod) {
        case InitMethod::random:
            init_centroids_rand();
            break;
        case InitMethod::plus_plus:
            init_centroids_plus_plus();
            break;
    }
}

Kmeans::~Kmeans() {
    delete generator;
    delete[] h_points;
    delete[] B;
    delete[] clusters;
    delete[] clusters_len;
    delete[] V_vals;
    delete[] V_colinds;
    delete[] V_rowptrs;
    delete[] distances;
    delete[] points_row_norms;
    delete[] centroids_row_norms;
    delete[] z_vals;
}

template <typename KernelType>
void Kmeans::init_kernel_matrix() {
    init_kernel_mtx_cpu<KernelType>(n, k, d, h_points, B, level);
}

void Kmeans::compute_points_norms() {
    // Extract diagonal from B and scale
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        points_row_norms[i] = B[i * n + i] / -2.0;
    }
}

void Kmeans::compute_centroids_norms() {
    // Compute c_tilde using SpMV: c_tilde = -0.5 * V * z
    std::fill(centroids_row_norms, centroids_row_norms + k, 0.0);
    
    #pragma omp parallel for
    for (uint32_t i = 0; i < k; i++) {
        DATA_TYPE sum = 0.0;
        for (int32_t idx = V_rowptrs[i]; idx < V_rowptrs[i + 1]; idx++) {
            int32_t col = V_colinds[idx];
            sum += V_vals[idx] * z_vals[col];
        }
        centroids_row_norms[i] = -0.5 * sum;
        // Filter out zero norms
        if (centroids_row_norms[i] == 0) {
            centroids_row_norms[i] = std::numeric_limits<DATA_TYPE>::infinity();
        }
    }
}

void Kmeans::compute_v_matrix() {
    compute_v_sparse_csr_cpu(V_vals, V_colinds, V_rowptrs,
                            (uint32_t*)clusters, clusters_len,
                            n, k);
}

void Kmeans::init_centroids_rand() {
    // Assign each point to a cluster in round-robin fashion
    for (size_t i = 0; i < n; i++) {
        clusters[i] = i % k;
    }

    // Count cluster sizes
    std::fill(clusters_len, clusters_len + k, 0);
    for (size_t i = 0; i < n; i++) {
        clusters_len[clusters[i]]++;
    }

#if LOG
    std::cout << "CLUSTER LENS" << std::endl;
    for (uint32_t i = 0; i < k; i++) {
        std::cout << (1.0 / clusters_len[i]) << ",";
    }
    std::cout << std::endl;
#endif

    // Build V matrix
    if (level > 0) {
        compute_v_matrix();
    }
}

void Kmeans::init_centroids_plus_plus() {
    std::cout << "K-means++ initialization not fully implemented for CPU version" << std::endl;
    std::cout << "Falling back to random initialization" << std::endl;
    init_centroids_rand();
}

void Kmeans::compute_distances_naive() {
    // Temporary buffer for first step
    DATA_TYPE* tmp = new DATA_TYPE[n * k];
    std::fill(tmp, tmp + n * k, 0.0);

    // Sum points in each cluster
    sum_points_cpu(B, clusters, clusters_len, tmp, n, k);

    // Compute centroid norms
    std::fill(centroids_row_norms, centroids_row_norms + k, 0.0);
    sum_centroids_cpu(tmp, clusters, clusters_len, centroids_row_norms, n, k);

    // Final distances
    compute_distances_naive_cpu(B, centroids_row_norms, tmp, distances, n, k);

    // Reset centroid norms for next iteration
    std::fill(centroids_row_norms, centroids_row_norms + k, 0.0);

    delete[] tmp;
}

void Kmeans::compute_distances_optimized() {
    // Step 1: D = V * B (sparse-dense multiplication)
    compute_distances_spmm_cpu(n, k, B, V_vals, V_colinds, V_rowptrs, distances);

    // Step 2: Initialize z vector with diagonal elements of D corresponding to cluster assignments
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        uint32_t cluster = clusters[i];
        z_vals[i] = distances[cluster + i * k];
    }

    // Step 3: Compute centroid norms using SpMV
    compute_centroids_norms();

    // Step 4: Add norms to distance matrix
    #pragma omp parallel for collapse(2)
    for (size_t j = 0; j < n; j++) {
        for (uint32_t i = 0; i < k; i++) {
            distances[i + j * k] += centroids_row_norms[i];
        }
    }
}

uint64_t Kmeans::run(uint64_t maxiter, bool check_converged) {
    uint64_t converged = maxiter;
    uint64_t iter = 0;

#if LOG
    std::ofstream centroids_out;
    centroids_out.open("centroids-cpu.out");
#endif

    // Main iteration loop
    while (iter++ < maxiter) {
        
        // STEP 1: Compute distances
        if (level >= OPT_MTX) {
            compute_distances_optimized();
        } else if (level == NAIVE_MTX) {
            // Use matrix-based method (not fully optimized)
            compute_distances_optimized();
        } else if (level == NAIVE_GPU) {
            compute_distances_naive();
        }

#if LOG
        centroids_out << "BEGIN DISTANCES ITER " << (iter - 1) << std::endl;
        for (size_t i = 0; i < n; i++) {
            for (uint32_t j = 0; j < k; j++) {
                centroids_out << distances[j + i * k] << ",";
            }
            centroids_out << std::endl;
        }
        centroids_out << "END DISTANCES ITER " << (iter - 1) << std::endl;
#endif

        // STEP 2: Assign points to nearest clusters
        clusters_argmin_cpu(n, k, distances, (uint32_t*)clusters, clusters_len, false);

        // STEP 3: Compute new centroids (update V matrix)
        if (level > 0) {
            compute_v_matrix();
        }

        // STEP 4: Compute objective score
        score = 0.0;
        #pragma omp parallel for reduction(+:score)
        for (size_t i = 0; i < n; i++) {
            uint32_t cluster = clusters[i];
            score += distances[cluster + i * k];
        }

        // STEP 5: Check convergence
        if (iter == maxiter) {
            break;
        }

        if (check_converged && (iter > 1) && (std::abs(score - last_score) < tol)) {
            converged = iter;
            break;
        }

        last_score = score;

#if LOG
        centroids_out << "Score: " << score << std::endl;
        centroids_out << "END ITERATION " << (iter - 1) << std::endl;
#endif
    }

    // Copy final cluster assignments to output
    for (size_t i = 0; i < n; i++) {
        h_points_clusters[i] = clusters[i];
        points[i]->setCluster(clusters[i]);
    }

#if LOG
    centroids_out.close();
#endif

#ifdef LOG_LABELS
    std::ofstream labels;
    labels.open("labels-cpu.out");
    for (size_t i = 0; i < n; i++) {
        labels << h_points_clusters[i] << std::endl;
    }
    labels.close();
#endif

#if PROFILE_MEMORY
    // Memory profiling on CPU
    size_t total_mem = 0;
    total_mem += n * d * sizeof(DATA_TYPE);  // points
    total_mem += n * n * sizeof(DATA_TYPE);  // kernel matrix
    total_mem += k * n * sizeof(DATA_TYPE);  // distances
    total_mem += n * sizeof(DATA_TYPE);      // V_vals
    std::cout << "MEMORY FOOTPRINT: " << (total_mem / 1e6) << " MB" << std::endl;
#endif

    return converged;
}