
//@HEADER
// ***************************************************
//
// HPCG: High Performance Conjugate Gradient Benchmark
//
// Contact:
// Michael A. Heroux ( maherou@sandia.gov)
// Jack Dongarra     (dongarra@eecs.utk.edu)
// Piotr Luszczek    (luszczek@eecs.utk.edu)
//
// ***************************************************
//@HEADER

/*!
 @file main.cpp

 HPCG routine
 */

// Main routine of a program that calls the HPCG conjugate gradient
// solver to solve the problem, and then prints results.

#ifndef HPCG_NO_MPI
#include <mpi.h>
#endif

#include <fstream>
#include <iostream>
#include <cstdlib>
#ifdef HPCG_DETAILED_DEBUG
using std::cin;
#endif
using std::endl;

#include <vector>

#include "hpcg.hpp"

#include "CheckAspectRatio.hpp"
#include "GenerateGeometry.hpp"
#include "GenerateProblem.hpp"
#include "GenerateCoarseProblem.hpp"
#include "SetupHalo.hpp"
#include "CheckProblem.hpp"
#include "ExchangeHalo.hpp"
#include "OptimizeProblem.hpp"
#include "WriteProblem.hpp"
#include "ReportResults.hpp"
#include "mytimer.hpp"
#include "ComputeSPMV_ref.hpp"
#include "ComputeMG_ref.hpp"
#include "ComputeResidual.hpp"
#include "CG.hpp"
#include "CG_ref.hpp"
#include "Geometry.hpp"
#include "SparseMatrix.hpp"
#include "Vector.hpp"
#include "CGData.hpp"
#include "TestCG.hpp"
#include "TestSymmetry.hpp"
#include "TestNorms.hpp"
#include <stdio.h>
#include <string>

#include <ccutils/timers.hpp> 
#include <ccutils/mpi/mpi_timers.hpp>
#include <ccutils/mpi/mpi_macros.hpp>

//TODO: transform this into macros in ccutils
bool collect_info = false;
extern std::vector<float> __timer_vals_cg_times;
extern std::vector<float> __timer_vals_dotp_allreduce_times;
extern std::vector<float> __timer_vals_dotp_times;
extern std::vector<float> __timer_vals_halo_times;
extern std::vector<float> __timer_vals_spmv_times;
extern std::vector<float> __timer_vals_mg_times;
extern std::vector<float> __timer_vals_waxpby_times;
extern std::vector<std::string> halo_kernel_call;
extern std::vector<std::vector<size_t>> halo_msg_size;

#ifdef HPCG_METRICS
  #define PRINT_SEP(title) if (rank == 0) { HPCG_fout << "==== " << title << " ====\n"; }
#else
  #define PRINT_SEP(title)
#endif

/*!
  Main driver program: Construct synthetic problem, run V&V tests, compute benchmark parameters, run benchmark, report results.

  @param[in]  argc Standard argument count.  Should equal 1 (no arguments passed in) or 4 (nx, ny, nz passed in)
  @param[in]  argv Standard argument array.  If argc==1, argv is unused.  If argc==4, argv[1], argv[2], argv[3] will be interpreted as nx, ny, nz, resp.

  @return Returns zero on success and a non-zero value otherwise.

*/
int main(int argc, char * argv[]) {

#ifndef HPCG_NO_MPI
  MPI_Init(&argc, &argv);
  CCUTILS_MPI_INIT
#endif

  HPCG_Params params;

  HPCG_Init(&argc, &argv, params);

  char host_name[MPI_MAX_PROCESSOR_NAME];
	int namelen;
	MPI_Get_processor_name(host_name,&namelen);
	CCUTILS_MPI_SECTION_DEF(node_names, "Processes node names")
	CCUTILS_MPI_ALL_PRINT_NAMED(node_names, fprintf(fp, "%s\n", host_name);)
	CCUTILS_MPI_SECTION_END(node_names)

  // Check if QuickPath option is enabled.
  // If the running time is set to zero, we minimize all paths through the program
  bool quickPath = (params.runningTime==0);

  int size = params.comm_size, rank = params.comm_rank; // Number of MPI processes, My process ID

#ifdef HPCG_DETAILED_DEBUG
  if (size < 100 && rank==0) HPCG_fout << "Process "<<rank<<" of "<<size<<" is alive with " << params.numThreads << " threads." <<endl;

  // if (rank==0) {
  //   char c;
  //   std::cout << "Press key to continue"<< std::endl;
  //   std::cin.get(c);
  // }
#ifndef HPCG_NO_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif
#endif

  local_int_t nx,ny,nz;
  nx = (local_int_t)params.nx;
  ny = (local_int_t)params.ny;
  nz = (local_int_t)params.nz;
  int ierr = 0;  // Used to check return codes on function calls

  ierr = CheckAspectRatio(0.125, nx, ny, nz, "local problem", rank==0);
  if (ierr)
    return ierr;

  /////////////////////////
  // Problem setup Phase //
  /////////////////////////

#ifdef HPCG_DEBUG
  double t1 = mytimer();
#endif

  // Construct the geometry and linear system
  Geometry * geom = new Geometry;
  GenerateGeometry(size, rank, params.numThreads, params.pz, params.zl, params.zu, nx, ny, nz, params.npx, params.npy, params.npz, geom);

  ierr = CheckAspectRatio(0.125, geom->npx, geom->npy, geom->npz, "process grid", rank==0);
  if (ierr)
    return ierr;
  PRINT_SEP("Geometry Constructed")

  // Use this array for collecting timing information
  std::vector< double > times(10,0.0);

  double setup_time = mytimer();

  SparseMatrix A;
  InitializeSparseMatrix(A, geom);
  PRINT_SEP("Matrix Initialized")

  Vector b, x, xexact;
  GenerateProblem(A, &b, &x, &xexact);
  PRINT_SEP("Problem Generated")
  SetupHalo(A);
  PRINT_SEP("Halo Setup")
  int numberOfMgLevels = 4; // Number of levels including first
  SparseMatrix * curLevelMatrix = &A;
  for (int level = 1; level< numberOfMgLevels; ++level) {
    GenerateCoarseProblem(*curLevelMatrix);
    curLevelMatrix = curLevelMatrix->Ac; // Make the just-constructed coarse grid the next level
  }
  PRINT_SEP("Coarse Problem Generated")

  setup_time = mytimer() - setup_time; // Capture total time of setup
  times[9] = setup_time; // Save it for reporting

  // {
  //   int n_neigh = A.numberOfSendNeighbors;
  //   std::string neighborInfo;
  //   neighborInfo.reserve(n_neigh * 64);
  //   char buf[64];
  //   for (int i = 0; i < n_neigh; ++i) {
  //     local_int_t n_send = A.sendLength[i];
  //     local_int_t n_recv = A.receiveLength[i];
  //     int ret = snprintf(buf, sizeof(buf), "%d:%lld,%lld|",
  //                        A.neighbors[i], (long long)n_send, (long long)n_recv);
  //     if (ret > 0) neighborInfo.append(buf, (size_t)ret);
  //   }
  //   // neighborInfo now holds the formatted send/recv info for all neighbors
  //   printf("[ExchangeHalo] Rank %d|%s\n", rank, neighborInfo.c_str());
  //   MPI_Barrier(MPI_COMM_WORLD);
  // }

  curLevelMatrix = &A;
  Vector * curb = &b;
  Vector * curx = &x;
  Vector * curxexact = &xexact;
  for (int level = 0; level< numberOfMgLevels; ++level) {
     CheckProblem(*curLevelMatrix, curb, curx, curxexact);
     curLevelMatrix = curLevelMatrix->Ac; // Make the nextcoarse grid the next level
     curb = 0; // No vectors after the top level
     curx = 0;
     curxexact = 0;
  }
  PRINT_SEP("Problem Checked")


  CGData data;
  InitializeSparseCGData(A, data);



  ////////////////////////////////////
  // Reference SpMV+MG Timing Phase //
  ////////////////////////////////////

  // Call Reference SpMV and MG. Compute Optimization time as ratio of times in these routines

  local_int_t nrow = A.localNumberOfRows;
  local_int_t ncol = A.localNumberOfColumns;

  Vector x_overlap, b_computed;
  InitializeVector(x_overlap, ncol); // Overlapped copy of x vector
  InitializeVector(b_computed, nrow); // Computed RHS vector


  // Record execution time of reference SpMV and MG kernels for reporting times
  // First load vector with random values
  FillRandomVector(x_overlap);
  PRINT_SEP("Concluded Initialization")

  int numberOfCalls = 10;
  if (quickPath) numberOfCalls = 1; //QuickPath means we do on one call of each block of repetitive code
  double t_begin = mytimer();
  double dummy = 0.0;
  for (int i=0; i< numberOfCalls; ++i) {
    ierr = ComputeSPMV_ref(A, x_overlap, b_computed, dummy); // b_computed = A*x_overlap
    if (ierr) HPCG_fout << "Error in call to SpMV: " << ierr << ".\n" << endl;
    ierr = ComputeMG_ref(A, b_computed, x_overlap); // b_computed = Minv*y_overlap
    if (ierr) HPCG_fout << "Error in call to MG: " << ierr << ".\n" << endl;
  }
  times[8] = (mytimer() - t_begin)/((double) numberOfCalls);  // Total time divided by number of calls.
#ifdef HPCG_DEBUG
  if (rank==0) HPCG_fout << "Total SpMV+MG timing phase execution time in main (sec) = " << mytimer() - t1 << endl;
#endif
  PRINT_SEP("Tested Runtime of SpMV+MG")

  ///////////////////////////////
  // Reference CG Timing Phase //
  ///////////////////////////////

#ifdef HPCG_DEBUG
  t1 = mytimer();
#endif
  int global_failure = 0; // assume all is well: no failures

  int niters = 0;
  int totalNiters_ref = 0;
  double normr = 0.0;
  double normr0 = 0.0;
  int refMaxIters = 50;
  numberOfCalls = 1; // Only need to run the residual reduction analysis once

  // Compute the residual reduction for the natural ordering and reference kernels
  std::vector< double > ref_times(9,0.0);
  double tolerance = 0.0; // Set tolerance to zero to make all runs do maxIters iterations
  int err_count = 0;
  for (int i=0; i< numberOfCalls; ++i) {
    ZeroVector(x);
    ierr = CG_ref( A, data, b, x, refMaxIters, tolerance, niters, normr, normr0, &ref_times[0], true);
    if (ierr) ++err_count; // count the number of errors in CG
    totalNiters_ref += niters;
  }
  if (rank == 0 && err_count) HPCG_fout << err_count << " error(s) in call(s) to reference CG." << endl;
  double refTolerance = normr / normr0;

  // Call user-tunable set up function.
  double t7 = mytimer();
  OptimizeProblem(A, data, b, x, xexact);
  t7 = mytimer() - t7;
  times[7] = t7;
#ifdef HPCG_DEBUG
  if (rank==0) HPCG_fout << "Total problem setup time in main (sec) = " << mytimer() - t1 << endl;
#endif

#ifdef HPCG_DETAILED_DEBUG
  if (geom->size == 1) WriteProblem(*geom, A, b, x, xexact);
#endif
  PRINT_SEP("Tested Runtime of CG")


  //////////////////////////////
  // Validation Testing Phase //
  //////////////////////////////

#ifdef HPCG_DEBUG
  t1 = mytimer();
#endif
  TestCGData testcg_data;
  testcg_data.count_pass = testcg_data.count_fail = 0;
  TestCG(A, data, b, x, testcg_data);

  TestSymmetryData testsymmetry_data;
  TestSymmetry(A, b, xexact, testsymmetry_data);

#ifdef HPCG_DEBUG
  if (rank==0) HPCG_fout << "Total validation (TestCG and TestSymmetry) execution time in main (sec) = " << mytimer() - t1 << endl;
#endif
  PRINT_SEP("Tested Correctness")

#ifdef HPCG_DEBUG
  t1 = mytimer();
#endif

  //////////////////////////////
  // Optimized CG Setup Phase //
  //////////////////////////////

  niters = 0;
  normr = 0.0;
  normr0 = 0.0;
  err_count = 0;
  int tolerance_failures = 0;

  int optMaxIters = 10*refMaxIters;
  int optNiters = refMaxIters;
  double opt_worst_time = 0.0;

  std::vector< double > opt_times(9,0.0);

  // Compute the residual reduction and residual count for the user ordering and optimized kernels.
  for (int i=0; i< numberOfCalls; ++i) {
    ZeroVector(x); // start x at all zeros
    double last_cummulative_time = opt_times[0];
    ierr = CG( A, data, b, x, optMaxIters, refTolerance, niters, normr, normr0, &opt_times[0], true, i);
    if (ierr) ++err_count; // count the number of errors in CG
    // Convergence check accepts an error of no more than 6 significant digits of relTolerance
    if (normr / normr0 > refTolerance * (1.0 + 1.0e-6)) ++tolerance_failures; // the number of failures to reduce residual

    // pick the largest number of iterations to guarantee convergence
    if (niters > optNiters) optNiters = niters;

    double current_time = opt_times[0] - last_cummulative_time;
    if (current_time > opt_worst_time) opt_worst_time = current_time;
  }

#ifndef HPCG_NO_MPI
// Get the absolute worst time across all MPI ranks (time in CG can be different)
  double local_opt_worst_time = opt_worst_time;
  MPI_Allreduce(&local_opt_worst_time, &opt_worst_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif


  if (rank == 0 && err_count) HPCG_fout << err_count << " error(s) in call(s) to optimized CG." << endl;
  if (tolerance_failures) {
    global_failure = 1;
    if (rank == 0)
      HPCG_fout << "Failed to reduce the residual " << tolerance_failures << " times." << endl;
  }
  PRINT_SEP("Optimized CG Setup")

  ///////////////////////////////
  // Optimized CG Timing Phase //
  ///////////////////////////////

  // Here we finally run the benchmark phase
  // The variable total_runtime is the target benchmark execution time in seconds

  double total_runtime = params.runningTime;
  int numberOfCgSets = int(total_runtime / opt_worst_time) + 1; // Run at least once, account for rounding

#ifdef HPCG_DEBUG
  if (rank==0) {
    HPCG_fout << "Projected running time: " << total_runtime << " seconds" << endl;
    HPCG_fout << "Number of CG sets: " << numberOfCgSets << endl;
  }
#endif

  /* This is the timed run for a specified amount of time. */

  optMaxIters = optNiters;
  double optTolerance = 0.0;  // Force optMaxIters iterations
  TestNormsData testnorms_data;
  testnorms_data.samples = numberOfCgSets;
  testnorms_data.values = new double[numberOfCgSets];

  PRINT_SEP("Beginning of Main Runs Loop")
  collect_info = true;

  #ifdef HPCG_METRICS
    std::string debugOutput;
    debugOutput += "[Halo Comm]<Rank " + std::to_string(A.geom->rank) + "> number of neighbors = " + std::to_string(A.numberOfSendNeighbors) + "\n";
    debugOutput += "Neighbor IDs: ";
    for (int i = 0; i < A.numberOfSendNeighbors; i++) {
      debugOutput += std::to_string(A.neighbors[i]) + " ";
    }
    debugOutput += "\nMsg sizes: ";
    for (int i = 0; i < A.numberOfSendNeighbors; i++) {
      debugOutput += std::to_string(A.sendLength[i]) + " ";
    }

    debugOutput += "\nAc Neighbor IDs: ";
    for (int i = 0; i < A.Ac->numberOfSendNeighbors; i++) {
      debugOutput += std::to_string(A.Ac->neighbors[i]) + " ";
    }
    debugOutput += "\nAc Msg sizes: ";
    for (int i = 0; i < A.Ac->numberOfSendNeighbors; i++) {
      debugOutput += std::to_string(A.Ac->sendLength[i]) + " ";
    }
    HPCG_fout << debugOutput << endl;
    MPI_Barrier(MPI_COMM_WORLD);
  #endif
  

  CCUTILS_MPI_SECTION_DEF(cg, "CG Benchmark Runs");

  for (int i=0; i< numberOfCgSets; ++i) {
    PRINT_SEP("Start of Run " << i)
    ZeroVector(x); // Zero out x
    ierr = CG( A, data, b, x, optMaxIters, optTolerance, niters, normr, normr0, &times[0], true, i);
    if (ierr) HPCG_fout << "Error in call to CG: " << ierr << ".\n" << endl;
    if (rank==0) HPCG_fout << "Call [" << i << "] Scaled Residual [" << normr/normr0 << "]" << endl;
    testnorms_data.values[i] = normr/normr0; // Record scaled residual from this run
    std::string key_iter = std::to_string(i);
    CCUTILS_SECTION_JSON_SUB_PUT(cg, key_iter, "cg_times", __timer_vals_cg_times);
    CCUTILS_SECTION_JSON_SUB_PUT(cg, key_iter, "dotp_allreduce", __timer_vals_dotp_allreduce_times);
    CCUTILS_SECTION_JSON_SUB_PUT(cg, key_iter, "dotp", __timer_vals_dotp_times);
    CCUTILS_SECTION_JSON_SUB_PUT(cg, key_iter, "spmv", __timer_vals_spmv_times);
    CCUTILS_SECTION_JSON_SUB_PUT(cg, key_iter, "mg", __timer_vals_mg_times);
    CCUTILS_SECTION_JSON_SUB_PUT(cg, key_iter, "waxpby", __timer_vals_waxpby_times);
    CCUTILS_SECTION_JSON_SUB_PUT(cg, key_iter, "exchange_halo", __timer_vals_halo_times);
    CCUTILS_SECTION_JSON_SUB_PUT(cg, key_iter, "halo_kernels", halo_kernel_call);
    CCUTILS_SECTION_JSON_SUB_PUT(cg, key_iter, "halo_msg_sizes", halo_msg_size);

    //clear vectors for next iteration
    std::vector<std::string>().swap(halo_kernel_call);
    std::vector<std::vector<size_t>>().swap(halo_msg_size);
    __timer_vals_cg_times.clear();
    __timer_vals_dotp_allreduce_times.clear();
    __timer_vals_dotp_times.clear();
    __timer_vals_spmv_times.clear();
    __timer_vals_mg_times.clear();
    __timer_vals_waxpby_times.clear();
    __timer_vals_halo_times.clear();

    PRINT_SEP("End of Run " << i)
  }

  // Compute difference between known exact solution and computed solution
  // All processors are needed here.
#ifdef HPCG_DEBUG
  double residual = 0;
  ierr = ComputeResidual(A.localNumberOfRows, x, xexact, residual);
  if (ierr) HPCG_fout << "Error in call to compute_residual: " << ierr << ".\n" << endl;
  if (rank==0) HPCG_fout << "Difference between computed and exact  = " << residual << ".\n" << endl;
#endif

  // Test Norm Results
  ierr = TestNorms(testnorms_data);

  PRINT_SEP("Reporting Results")
  
  CCUTILS_MPI_GLOBAL_JSON_PUT(cg, "world_size", size);
  CCUTILS_MPI_GLOBAL_JSON_PUT(cg, "cg_sets", numberOfCgSets);
  CCUTILS_MPI_GLOBAL_JSON_PUT(cg, "grid_size_nx", params.nx);
  CCUTILS_MPI_GLOBAL_JSON_PUT(cg, "grid_size_ny", params.ny);
  CCUTILS_MPI_GLOBAL_JSON_PUT(cg, "grid_size_nz", params.nz);

  CCUTILS_MPI_SECTION_END(cg);

  ////////////////////
  // Report Results //
  ////////////////////

  // Report results to YAML file
  CCUTILS_MPI_SECTION_DEF(hpcg_output, "Classic HPCG benchmark output")
  ReportResults(A, numberOfMgLevels, numberOfCgSets, refMaxIters, optMaxIters, &times[0], testcg_data, testsymmetry_data, testnorms_data, global_failure, quickPath);
  CCUTILS_MPI_SECTION_END(hpcg_output)
  
  PRINT_SEP("Cleanup")
  // Clean up
  DeleteMatrix(A); // This delete will recursively delete all coarse grid data
  DeleteCGData(data);
  DeleteVector(x);
  DeleteVector(b);
  DeleteVector(xexact);
  DeleteVector(x_overlap);
  DeleteVector(b_computed);
  delete [] testnorms_data.values;

  HPCG_Finalize();

  // Finish up
#ifndef HPCG_NO_MPI
  MPI_Finalize();
#endif
  return 0;
}
