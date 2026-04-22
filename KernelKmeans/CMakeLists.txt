cmake_minimum_required(VERSION 3.24.3)

project(Popcorn_Kernel_Kmeans VERSION 1.0.0 LANGUAGES CXX)

# Set "flang_rt.runtime" for the HCA cluster 
set(EXTRA_LIBS "" CACHE STRING "Extra libraries to link")

set(NOFORTRAN ON)
find_package(OpenBLAS REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(CCUTILS_ENABLE_CUDA OFF CACHE BOOL "Enable CUDA support in ccutils")
set(CCUTILS_ENABLE_MPI OFF CACHE BOOL "Enable MPI support in ccutils")
set(ccutils_DIR "~/.local/ccutils/install/lib/cmake/ccutils/")
message(STATUS "Including CCUTILS from: ${ccutils_DIR}")
# message(STATUS "CCUTILS include dir: $ENV{CCUTILS_INCLUDE}")
find_package(ccutils REQUIRED)

message(STATUS "OpenBLAS_LIBRARIES: ${OpenBLAS_LIBRARIES}")


# OpenBLAS + OpenMP target
add_executable(popcornkmeans_openblas
  ./src/kernels_openblas.cpp
  ./src/kmeans.cpp
  ./src/main.cpp
)
target_link_libraries(popcornkmeans_openblas PRIVATE ${OpenBLAS_LIBRARIES} ccutils::ccutils ${EXTRA_LIBS})
target_include_directories(popcornkmeans_openblas PRIVATE ${OpenBLAS_INCLUDE_DIRS} ./include)


# OpenMP target: kernels are implemented "manually"
add_executable(popcornkmeans_openmp
  ./src/kernels_openmp.cpp
  ./src/kmeans.cpp
  ./src/main.cpp
)
target_link_libraries(popcornkmeans_openmp PRIVATE ccutils::ccutils ${EXTRA_LIBS})
target_include_directories(popcornkmeans_openmp PRIVATE ./include)



