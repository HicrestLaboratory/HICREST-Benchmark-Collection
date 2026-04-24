#include <chrono>
#include <iostream>
#include <fstream>
#include <float.h>

#include <ccutils/timers.hpp>
#include <ccutils/macros.hpp>
#include <ccutils/colors.h>

#include "../include/common.h"
#include "../include/input_parser.hpp"
#include "../include/utils.hpp"

#include "kmeans.h"

#ifdef _OPENMP
  #include <omp.h>
#endif

using namespace std;

// Global timers
CCUTILS_CPU_TIMER_DEF(distances_compute)
CCUTILS_CPU_TIMER_DEF(argmin_assign)
CCUTILS_CPU_TIMER_DEF(v_matrix_update)
CCUTILS_CPU_TIMER_DEF(score_compute)
CCUTILS_CPU_TIMER_DEF(total_iteration)

int main(int argc, char **argv) {
  uint32_t d, k, runs;
  size_t n, maxiter;
  string out_file;
  float tol;
  int *seed = NULL;
  InputParser<DATA_TYPE> *input = NULL;
  bool check_converged;
  string init_method_str, kernel_str;

  parse_input_args(argc, argv, 
                   &d, &n, &k, 
                   &maxiter, out_file, 
                   &tol, &runs, &seed, &input,
                   &check_converged,
                   init_method_str,
                   kernel_str);

  #if DEBUG_INPUT_DATA
    cout << "Points" << endl << *input << endl;
  #endif

  CCUTILS_CPU_TIMER_DEF(init)

  Kmeans::InitMethod init_method;
  if (init_method_str.compare("random") == 0) {
    init_method = Kmeans::InitMethod::random;
  } else if (init_method_str.compare("plus_plus") == 0) {
    init_method = Kmeans::InitMethod::plus_plus;
  } else {
    printf("Invalid init method: %s\n", init_method_str.c_str());
    exit(1);
  }

  Kmeans::Kernel kernel;
  if (kernel_str.compare("linear") == 0) {
    kernel = Kmeans::Kernel::linear;
  } else if (kernel_str.compare("polynomial") == 0) {
    kernel = Kmeans::Kernel::polynomial;
  } else if (kernel_str.compare("sigmoid") == 0) {
    kernel = Kmeans::Kernel::sigmoid;
  } else {
    printf("Invalid kernel: %s\n", kernel_str.c_str());
    exit(1);
  }

  // Set number of OpenMP threads
  #ifdef _OPENMP
  int num_threads = omp_get_max_threads();
  printf("Using OpenMP with %d threads\n", num_threads);
  #endif

  double tot_score = 0;
  double min_score = DBL_MAX;
  double max_score = 0.0;
  for (uint32_t i = 0; i < runs; i++) {
    printf("=== Run %u ===\n", i);

    CCUTILS_CPU_TIMER_START(init)
    Kmeans kmeans(n, d, k, tol, seed, input->get_dataset(), init_method, kernel);
    CCUTILS_CPU_TIMER_STOP(init)

    uint64_t iters = kmeans.run(maxiter, check_converged);

    printf("Iterations: %lu\n", iters);
    printf("Objective score (ideal is 0.0): %lf\n", kmeans.get_score());
    tot_score += kmeans.get_score();
    min_score = min(min_score, kmeans.get_score());
    max_score = max(max_score, kmeans.get_score());
  }

  size_t total_mem = 0;
  total_mem += n * d * sizeof(DATA_TYPE);  // points
  total_mem += n * n * sizeof(DATA_TYPE);  // kernel matrix
  total_mem += k * n * sizeof(DATA_TYPE);  // distances
  total_mem += n * sizeof(DATA_TYPE);      // V_vals

  CCUTILS_SECTION_DEF(config, "config")
  CCUTILS_SECTION_JSON_PUT(config, "n", n)
  CCUTILS_SECTION_JSON_PUT(config, "d", d)
  CCUTILS_SECTION_JSON_PUT(config, "k", k)
  CCUTILS_SECTION_JSON_PUT(config, "runs", runs)
  CCUTILS_SECTION_JSON_PUT(config, "maxiter", maxiter)
  CCUTILS_SECTION_JSON_PUT(config, "seed", *seed)
  CCUTILS_SECTION_JSON_PUT(config, "init_method", init_method_str)
  CCUTILS_SECTION_JSON_PUT(config, "kernel", kernel_str)
  CCUTILS_SECTION_JSON_PUT(config, "total_mem_bytes", total_mem)
  CCUTILS_SECTION_JSON_PUT(config, "min_score", min_score)
  CCUTILS_SECTION_JSON_PUT(config, "max_score", max_score)
  CCUTILS_SECTION_JSON_PUT(config, "avg_score", tot_score / runs)
  CCUTILS_SECTION_END(config);

  CCUTILS_SECTION_DEF(timers, "timers [millis]")
  CCUTILS_SECTION_JSON_PUT(timers, "init", CCUTILS_TIMER_VALUES(init));
  CCUTILS_SECTION_JSON_PUT(timers, "distances_compute", CCUTILS_TIMER_VALUES(distances_compute));
  CCUTILS_SECTION_JSON_PUT(timers, "argmin_assign", CCUTILS_TIMER_VALUES(argmin_assign));
  CCUTILS_SECTION_JSON_PUT(timers, "v_matrix_update", CCUTILS_TIMER_VALUES(v_matrix_update));
  CCUTILS_SECTION_JSON_PUT(timers, "score_compute", CCUTILS_TIMER_VALUES(score_compute));
  CCUTILS_SECTION_JSON_PUT(timers, "total", CCUTILS_TIMER_VALUES(total_iteration));
  CCUTILS_SECTION_END(timers);

  if (strcmp(out_file.c_str(), "None") != 0) {
    ofstream fout(out_file);
    input->dataset_to_csv(fout);
    fout.close();
  }
  
  delete seed;
  return 0;
}