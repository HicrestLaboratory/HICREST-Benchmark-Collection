
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
 @file ComputeSPMV_ref.cpp

 HPCG routine
 */

#include "ComputeSPMV_ref.hpp"

#ifndef HPCG_NO_MPI
#include "ExchangeHalo.hpp"
#endif

#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif
#include <cassert>
#include <vector>
#include <string>

#include "mytimer.hpp"
#define TICK()  t0 = mytimer() //!< record current time in 't0'
#define TOCK(t) t += mytimer() - t0 //!< store time difference in 't' using time in 't0'

#include <ccutils/mpi/mpi_timers.hpp>
#include <ccutils/mpi/mpi_macros.hpp>
extern bool collect_info;
extern bool preconditioning;
CCUTILS_MPI_TIMER_DEF(spmv_times)

#define CCUTILS_MPI_TIMER_STOP_CONDITIONAL(timer) \
  CCUTILS_MPI_TIMER_STOP(timer)           \
  if(!collect_info)                       \
    __timer_vals_##timer.pop_back();

/*!
  Routine to compute matrix vector product y = Ax where:
  Precondition: First call exchange_externals to get off-processor values of x

  This is the reference SPMV implementation.  It CANNOT be modified for the
  purposes of this benchmark.

  @param[in]  A the known system matrix
  @param[in]  x the known vector
  @param[out] y the On exit contains the result: Ax.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeSPMV
*/
int ComputeSPMV_ref( const SparseMatrix & A, Vector & x, Vector & y, double &halo_timer) {

  assert(x.localLength>=A.localNumberOfColumns); // Test vector lengths
  assert(y.localLength>=A.localNumberOfRows);

#ifndef HPCG_NO_MPI
  //double t0 = 0.0;
  ExchangeHalo(A,x, "SPMV");
#endif
  CCUTILS_MPI_TIMER_START(spmv_times)
  const double * const xv = x.values;
  double * const yv = y.values;
  const local_int_t nrow = A.localNumberOfRows;
#ifndef HPCG_NO_OPENMP
  #pragma omp parallel for
#endif
  for (local_int_t i=0; i< nrow; i++)  {
    double sum = 0.0;
    const double * const cur_vals = A.matrixValues[i];
    const local_int_t * const cur_inds = A.mtxIndL[i];
    const int cur_nnz = A.nonzerosInRow[i];

    for (int j=0; j< cur_nnz; j++)
      sum += cur_vals[j]*xv[cur_inds[j]];
    yv[i] = sum;
  }
  CCUTILS_MPI_TIMER_STOP_CONDITIONAL(spmv_times);
  return 0;
}
