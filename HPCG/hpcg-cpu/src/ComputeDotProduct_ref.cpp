
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
 @file ComputeDotProduct_ref.cpp

 HPCG routine
 */

#ifndef HPCG_NO_MPI
#include <mpi.h>
#include "mytimer.hpp"
#endif
#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif
#include <cassert>
#include "ComputeDotProduct_ref.hpp"

#include <ccutils/mpi/mpi_timers.hpp>

//TODO: add the ifdef CCUTILS_TIMERS around the MPI_TIMER_DEF
CCUTILS_MPI_TIMER_DEF(dotp_allreduce_times)

extern bool collect_info;

#define MPI_TIMER_STOP_CONDITIONAL(timer) \
  CCUTILS_MPI_TIMER_STOP(timer)           \
  if(!collect_info)                       \
    __timer_vals_##timer.pop_back();
    
/*!
  Routine to compute the dot product of two vectors where:

  This is the reference dot-product implementation.  It _CANNOT_ be modified for the
  purposes of this benchmark.

  @param[in] n the number of vector elements (on this processor)
  @param[in] x, y the input vectors
  @param[in] result a pointer to scalar value, on exit will contain result.
  @param[out] time_allreduce the time it took to perform the communication between processes

  @return returns 0 upon success and non-zero otherwise

  @see ComputeDotProduct
*/
int ComputeDotProduct_ref(const local_int_t n, const Vector & x, const Vector & y,
    double & result, double & time_allreduce) {
  assert(x.localLength>=n); // Test vector lengths
  assert(y.localLength>=n);

  double local_result = 0.0;
  double * xv = x.values;
  double * yv = y.values;
  if (yv==xv) {
#ifndef HPCG_NO_OPENMP
    #pragma omp parallel for reduction (+:local_result)
#endif
    for (local_int_t i=0; i<n; i++) local_result += xv[i]*xv[i];
  } else {
#ifndef HPCG_NO_OPENMP
    #pragma omp parallel for reduction (+:local_result)
#endif
    for (local_int_t i=0; i<n; i++) local_result += xv[i]*yv[i];
  }

#ifndef HPCG_NO_MPI
  // Use MPI's reduce function to collect all partial sums
  #ifdef USE_CCUTILS_TIMERS
    CCUTILS_MPI_TIMER_START(dotp_allreduce_times)
  #else
  double t0 = mytimer();
  #endif
  double global_result = 0.0;
  MPI_Allreduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM,
      MPI_COMM_WORLD);
  result = global_result;

  #ifdef USE_CCUTILS_TIMERS
    MPI_TIMER_STOP_CONDITIONAL(dotp_allreduce_times)
  #else
  time_allreduce += mytimer() - t0;
  #endif
#else
  time_allreduce += 0.0;
  result = local_result;
#endif

  return 0;
}
