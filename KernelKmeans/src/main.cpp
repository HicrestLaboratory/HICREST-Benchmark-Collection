#include <chrono>
#include <iostream>
#include <fstream>

#include <ccutils/timers.hpp>
#include <ccutils/colors.h>

#include "../include/common.h"
#include "../include/input_parser.hpp"
#include "../include/utils.hpp"

#include "kmeans.h"

#ifdef _OPENMP
  #include <omp.h>
#endif

using namespace std;

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

  CCUTILS_CPU_TIMER_DEF(kernel_kmeans)
  CCUTILS_CPU_TIMER_DEF(init)
  double score = 0;

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

  for (uint32_t i = 0; i < runs; i++) {
    CCUTILS_CPU_TIMER_START(init)
    Kmeans kmeans(n, d, k, tol, seed, input->get_dataset(), init_method, kernel);
    CCUTILS_CPU_TIMER_STOP(init)

    CCUTILS_CPU_TIMER_START(kernel_kmeans)
    uint64_t iters = kmeans.run(maxiter, check_converged);
    CCUTILS_CPU_TIMER_STOP(kernel_kmeans)

    printf("=== Run %u ===\n", i);
    CCUTILS_TIMER_PRINT_LAST(kernel_kmeans)
    printf("Iterations: %lu\n", iters);
    printf("Objective score: %lf\n", kmeans.get_score());
    score += kmeans.get_score();
  }

  printf("=== Summary ===\n");
  if (runs > 1) {
    printf("The first run is excluded (warmup)\n");
    CCUTILS_TIMER_PRINT_EXCLUDING_FIRST_N(kernel_kmeans,1)
  } else {
    CCUTILS_TIMER_PRINT(kernel_kmeans)
  }
  printf("Average Score: %lf\n", score / runs);

  if (strcmp(out_file.c_str(), "None") != 0) {
    ofstream fout(out_file);
    input->dataset_to_csv(fout);
    fout.close();
  }
  
  delete seed;
  return 0;
}