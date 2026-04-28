#include "kmeans.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <omp.h>
#include <ccutils/timers.hpp>

// Import global timers
CCUTILS_CPU_TIMER_IMPORT(distances_compute)
CCUTILS_CPU_TIMER_IMPORT(argmin_assign)
CCUTILS_CPU_TIMER_IMPORT(v_matrix_update)
CCUTILS_CPU_TIMER_IMPORT(score_compute)
CCUTILS_CPU_TIMER_IMPORT(total_iteration)

/**
 * ============================================================================
 * CONSTRUCTOR
 * ============================================================================
 */
Kmeans::Kmeans(const size_t _n, const uint32_t _d, const uint32_t _k,
               const float _tol, const int* seed,
               Point<float>** _points,
               const InitMethod _initMethod,
               const Kernel _kernel)
    : n(_n), d(_d), k(_k), tol(_tol),
      initMethod(_initMethod),
      kernel(_kernel),
      score(0.0),
      last_score(0.0) {

    // ────────────────────────────────────────────────────────────────
    // Initialize random generator
    // ────────────────────────────────────────────────────────────────
    
    if (seed != nullptr) {
        generator = new std::mt19937(*seed);
    } else {
        std::random_device rd;
        generator = new std::mt19937(rd());
    }

    // ────────────────────────────────────────────────────────────────
    // Allocate matrices
    // ────────────────────────────────────────────────────────────────
    
    K.resize(n * n);
    p_norms.resize(n);
    distances.resize(n * k);
    clusters.resize(n);
    clusters_len.resize(k);
    
    // Allocate CSR sparse matrix V (k×n with exactly n non-zeros)
    V_vals.resize(n);
    V_colinds.resize(n);
    V_rowptrs.resize(k + 1);

    // ────────────────────────────────────────────────────────────────
    // Copy points from input Point objects to flat array
    // ────────────────────────────────────────────────────────────────
    
    points.resize(n * d);
    for (size_t i = 0; i < n; ++i) {
        for (uint32_t j = 0; j < d; ++j) {
            points[i * d + j] = _points[i]->get(j);
        }
    }

#if LOG
    std::ofstream points_out;
    points_out.open("points-cpu.out");
    for (size_t i = 0; i < n; i++) {
        for (uint32_t j = 0; j < d; j++) {
            points_out << points[i * d + j] << ",";
        }
        points_out << std::endl;
    }
    points_out.close();
#endif

    // ────────────────────────────────────────────────────────────────
    // Initialize kernel matrix (done once before main loop)
    // ────────────────────────────────────────────────────────────────
    
    std::cout << "Initializing kernel matrix (" << n << "×" << n 
              << ") with kernel type: ";
    switch (kernel) {
        case Kernel::linear:
            std::cout << "linear" << std::endl;
            break;
        case Kernel::polynomial:
            std::cout << "polynomial" << std::endl;
            break;
        case Kernel::sigmoid:
            std::cout << "sigmoid" << std::endl;
            break;
    }
    
    initialize_kernel_matrix();

#ifdef LOG_KERNEL
    std::ofstream kernel_out;
    kernel_out.open("kernel-cpu.out");
    for (size_t i = 0; i < std::min(n, size_t(10)); i++) {
        for (size_t j = 0; j < n; j++) {
            kernel_out << K[i * n + j] << ",";
        }
        kernel_out << std::endl;
    }
    kernel_out.close();
#endif

    // ────────────────────────────────────────────────────────────────
    // Initialize clusters
    // ────────────────────────────────────────────────────────────────
    
    std::cout << "Initializing clusters..." << std::endl;
    initialize_clusters();
}

/**
 * ============================================================================
 * DESTRUCTOR
 * ============================================================================
 */
Kmeans::~Kmeans() {
    delete generator;
}

/**
 * ============================================================================
 * PRIVATE METHODS
 * ============================================================================
 */

void Kmeans::initialize_kernel_matrix() {
    // Step 1: Compute kernel matrix K = P̂·P̂^T
    // Uses optimized GEMM (BLAS on OpenBLAS, manual collapse(2) on OpenMP)
    init_kernel_mtx_gemm_cpu(n, d, points.data(), K.data());

    // Step 2: Apply kernel transformation based on kernel type
    switch (kernel) {
        case Kernel::linear:
            apply_linear_kernel_cpu(n, K.data());
            break;
        case Kernel::polynomial:
            apply_polynomial_kernel_cpu(n, K.data());
            break;
        case Kernel::sigmoid:
            apply_sigmoid_kernel_cpu(n, K.data());
            break;
    }

    // Step 3: Extract point norms from diagonal for P̃
    // P̃[i,j] = K[i,i] for all j (same value for all clusters)
    copy_diag_cpu(K.data(), p_norms.data(), n);
}

void Kmeans::initialize_clusters() {
    // Assign each point to a cluster
    switch (initMethod) {
        case InitMethod::random: {
            // Round-robin assignment: point i -> cluster (i % k)
            for (size_t i = 0; i < n; i++) {
                clusters[i] = i % k;
            }
            
            // Count cluster sizes
            std::fill(clusters_len.begin(), clusters_len.end(), 0);
            for (size_t i = 0; i < n; i++) {
                clusters_len[clusters[i]]++;
            }
            
#if LOG
            std::cout << "Initial cluster sizes: ";
            for (uint32_t i = 0; i < k; i++) {
                std::cout << clusters_len[i] << " ";
            }
            std::cout << std::endl;
#endif
            break;
        }

        case InitMethod::plus_plus: {
            std::cout << "K-means++ initialization not fully implemented on CPU" << std::endl;
            std::cout << "Falling back to random initialization" << std::endl;
            
            // Fall back to random initialization
            for (size_t i = 0; i < n; i++) {
                clusters[i] = i % k;
            }
            std::fill(clusters_len.begin(), clusters_len.end(), 0);
            for (size_t i = 0; i < n; i++) {
                clusters_len[clusters[i]]++;
            }
            break;
        }
    }

    // Build initial V matrix (cluster membership)
    compute_v_matrix();
}

void Kmeans::compute_distances() {
    // ────────────────────────────────────────────────────────────────
    // STEP 1: Compute distances using Popcorn formula
    // ────────────────────────────────────────────────────────────────
    // 
    // Formula: D = -2·K·V^T + P̃ + C̃
    //
    // Where:
    //   K = kernel matrix (n×n, constant across iterations)
    //   V = sparse cluster membership (k×n, CSR)
    //   P̃ = point norms ||φ(p_i)||² (n elements, constant)
    //   C̃ = centroid norms ||c_j||² (k elements, computed here)
    //   D = output distances (n×k, row-major)
    //
    // The function handles all four components internally:
    // 1. SpMM: E = -2·K·V^T
    // 2. Extract: z[i] = E[cluster[i], i]
    // 3. SpMV: c_norms = -0.5·V·z
    // 4. Assemble: D[i,j] = E[j,i] + P̃[i] + C̃[j]

    compute_distances_complete_cpu(
        n, k,                           // dimensions
        K.data(),                       // kernel matrix (constant)
        p_norms.data(),                 // point norms (constant)
        V_vals.data(),                  // CSR values
        V_colinds.data(),              // CSR column indices
        V_rowptrs.data(),              // CSR row pointers
        clusters.data(),               // cluster assignments
        distances.data());              // output distances
}

void Kmeans::compute_v_matrix() {
    // Build sparse matrix V in CSR format
    // V[j,i] = 1/|L_j| if point i is in cluster j, 0 otherwise
    //
    // This encodes which cluster each point belongs to and is used
    // to compute cluster-wise statistics in the distance formula.
    
    compute_v_sparse_csr_cpu(
        clusters.data(),                // input: cluster assignments
        clusters_len.data(),            // input: cluster sizes
        V_vals.data(),                  // output: CSR values
        V_colinds.data(),              // output: CSR column indices
        V_rowptrs.data(),              // output: CSR row pointers
        n, k);                         // dimensions
}

/**
 * ============================================================================
 * MAIN CLUSTERING LOOP
 * ============================================================================
 */
uint64_t Kmeans::run(uint64_t maxiter, bool check_converged) {
    uint64_t converged = maxiter;
    uint64_t iter = 0;

#if LOG
    std::ofstream distances_out;
    distances_out.open("distances-cpu.out");
    std::ofstream score_out;
    score_out.open("score-cpu.out");
#endif

    // ────────────────────────────────────────────────────────────────
    // MAIN ITERATION LOOP
    // ────────────────────────────────────────────────────────────────
    
    while (iter++ < maxiter) {
        CCUTILS_CPU_TIMER_START(total_iteration)
        
        // ────────────────────────────────────────────────────────────
        // STEP 1: Compute pairwise distances
        // ────────────────────────────────────────────────────────────
        // Most expensive step: O(n²) for SpMM kernel
        
        CCUTILS_CPU_TIMER_START(distances_compute)
        compute_distances();
        CCUTILS_CPU_TIMER_STOP(distances_compute)

#if LOG
        distances_out << "ITERATION " << (iter - 1) << std::endl;
        for (size_t i = 0; i < n; i++) {
            for (uint32_t j = 0; j < k; j++) {
                distances_out << distances[i * k + j] << ",";
            }
            distances_out << std::endl;
        }
#endif

        // ────────────────────────────────────────────────────────────
        // STEP 2: Assign points to nearest clusters
        // ────────────────────────────────────────────────────────────
        // For each point i, find cluster j = argmin_j D[i,j]
        // Update cluster assignments and sizes atomically
        
        CCUTILS_CPU_TIMER_START(argmin_assign)
        clusters_argmin_cpu(n, k, distances.data(),
                           clusters.data(), clusters_len.data());
        CCUTILS_CPU_TIMER_STOP(argmin_assign)

        // ────────────────────────────────────────────────────────────
        // STEP 3: Update cluster membership matrix
        // ────────────────────────────────────────────────────────────
        // Recompute V based on new cluster assignments
        // This is needed for the next iteration's distance computation
        
        CCUTILS_CPU_TIMER_START(v_matrix_update)
        compute_v_matrix();
        CCUTILS_CPU_TIMER_STOP(v_matrix_update)

        // ────────────────────────────────────────────────────────────
        // STEP 4: Compute objective score
        // ────────────────────────────────────────────────────────────
        // Sum of distances from each point to its assigned cluster
        // Used for convergence checking
        
        CCUTILS_CPU_TIMER_START(score_compute)
        score = 0.0;
        #pragma omp parallel for reduction(+:score) default(none) \
                shared(distances, clusters, n, k)
        for (size_t i = 0; i < n; i++) {
            uint32_t cluster = clusters[i];
            score += distances[i * k + cluster];
        }
        CCUTILS_CPU_TIMER_STOP(score_compute)

#if LOG
        score_out << "ITERATION " << (iter - 1) << ": " << score << std::endl;
#endif

        CCUTILS_CPU_TIMER_STOP(total_iteration)

        // ────────────────────────────────────────────────────────────
        // STEP 5: Check convergence
        // ────────────────────────────────────────────────────────────
        
        if (iter == maxiter) {
            // Reached maximum iterations
            break;
        }

        if (check_converged && (iter > 1)) {
            // Check if objective change is below tolerance
            double score_change = std::abs(score - last_score);
            if (score_change < tol) {
                converged = iter;
                std::cout << "Converged at iteration " << iter 
                          << " (score change: " << score_change << ")" << std::endl;
                break;
            }
        }

        last_score = score;
    }

    // ────────────────────────────────────────────────────────────────
    // Print timing statistics (optional)
    // ────────────────────────────────────────────────────────────────
    
    std::cout << "\n=== Iteration Timing Statistics ===" << std::endl;
    std::cout << "  Distances:     ";
    CCUTILS_TIMER_PRINT(distances_compute)
    std::cout << "  Argmin/Assign: ";
    CCUTILS_TIMER_PRINT(argmin_assign)
    std::cout << "  V Matrix:      ";
    CCUTILS_TIMER_PRINT(v_matrix_update)
    std::cout << "  Score Compute: ";
    CCUTILS_TIMER_PRINT(score_compute)
    std::cout << "  Total/Iter:    ";
    CCUTILS_TIMER_PRINT(total_iteration)

#if LOG
    distances_out.close();
    score_out.close();
#endif

    return converged;
}