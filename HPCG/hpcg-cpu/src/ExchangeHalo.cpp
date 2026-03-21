
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
 @file ExchangeHalo.cpp

 HPCG routine
 */

// Compile this routine only if running with MPI
#ifndef HPCG_NO_MPI
#include <mpi.h>
#include "Geometry.hpp"
#include "ExchangeHalo.hpp"
#include <cstdlib>

#include <ccutils/mpi/mpi_timers.hpp>
#include <ccutils/mpi/mpi_macros.hpp>
//TODO: add the ifdef CCUTILS_TIMERS around the MPI_TIMER_DEF
CCUTILS_MPI_TIMER_DEF(halo_times)
std::vector<std::string> halo_kernel_call;
std::vector<std::vector<size_t>> halo_msg_size;
extern bool collect_info;
extern bool preconditioning;

#define MPI_TIMER_STOP_CONDITIONAL(timer) \
  CCUTILS_MPI_TIMER_STOP(timer)           \
  if(!collect_info)                       \
    __timer_vals_##timer.pop_back();


/*!
  Communicates data that is at the border of the part of the domain assigned to this processor.

  @param[in]    A The known system matrix
  @param[inout] x On entry: the local vector entries followed by entries to be communicated; on exit: the vector with non-local entries updated by other processors
 */
void ExchangeHalo(const SparseMatrix & A, Vector & x, const char* kernel_name) {
  #ifdef USE_CCUTILS_TIMERS
    CCUTILS_MPI_TIMER_START(halo_times)
  #endif
  // Extract Matrix pieces

  local_int_t localNumberOfRows = A.localNumberOfRows;
  int num_neighbors = A.numberOfSendNeighbors;
  local_int_t * receiveLength = A.receiveLength;
  local_int_t * sendLength = A.sendLength;
  int * neighbors = A.neighbors;
  double * sendBuffer = A.sendBuffer;
  local_int_t totalToBeSent = A.totalToBeSent;
  local_int_t * elementsToSend = A.elementsToSend;

  double * const xv = x.values;

  int size, rank; // Number of MPI processes, My process ID
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  //
  //  first post receives, these are immediate receives
  //  Do not wait for result to come, will do that at the
  //  wait call below.
  //

  int MPI_MY_TAG = 99;

  MPI_Request * request = new MPI_Request[num_neighbors];

  //
  // Externals are at end of locals
  //
  double * x_external = (double *) xv + localNumberOfRows;

  // Post receives first
  // TODO: Thread this loop
  for (int i = 0; i < num_neighbors; i++) {
    local_int_t n_recv = receiveLength[i];
    MPI_Irecv(x_external, n_recv, MPI_DOUBLE, neighbors[i], MPI_MY_TAG, MPI_COMM_WORLD, request+i);
    x_external += n_recv;
  }


  //
  // Fill up send buffer
  //

  // TODO: Thread this loop
  for (local_int_t i=0; i<totalToBeSent; i++) sendBuffer[i] = xv[elementsToSend[i]];

  //
  // Send to each neighbor
  //

  // std::string dbg = "*Rank ";
  // dbg += std::to_string(rank);
  // dbg += "* send sizes: ";

  // TODO: Thread this loop
  for (int i = 0; i < num_neighbors; i++) {
    local_int_t n_send = sendLength[i];
    MPI_Send(sendBuffer, n_send, MPI_DOUBLE, neighbors[i], MPI_MY_TAG, MPI_COMM_WORLD);
    sendBuffer += n_send;
    // dbg += std::to_string(n_send);
    // dbg += " ";
  }

  //
  // Complete the reads issued above
  //

  // std::cerr << dbg << std::endl;
  
  MPI_Status status;
  // TODO: Thread this loop
  for (int i = 0; i < num_neighbors; i++) {
    if ( MPI_Wait(request+i, &status) ) {
      std::exit(-1); // TODO: have better error exit
    }
  }

  delete [] request;
  #ifdef USE_CCUTILS_TIMERS
    MPI_TIMER_STOP_CONDITIONAL(halo_times)
    
    if(collect_info) {
      halo_kernel_call.push_back(preconditioning ? "preconditioning_" + std::string(kernel_name) : kernel_name);
      std::vector<size_t> iter_msg_sizes(num_neighbors);
      for(int i=0; i<num_neighbors; i++)
        iter_msg_sizes[i] = static_cast<size_t>(sendLength[i]*sizeof(double)); // in bytes
      
      halo_msg_size.push_back(iter_msg_sizes);
    }
  #endif

  return;
}
#endif
// ifndef HPCG_NO_MPI
