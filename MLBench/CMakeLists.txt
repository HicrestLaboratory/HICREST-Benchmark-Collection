cmake_minimum_required(VERSION 3.14)
project(MLqualcosa CXX C ASM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# --- Backend selection (see mlops/ + llamacpp's own GGML_BLAS docs) --------
set(BACKEND "cpu" CACHE STRING "Compute backend: cpu | blas | blis")
set_property(CACHE BACKEND PROPERTY STRINGS cpu blas blis)
if(BACKEND STREQUAL "blas")
    set(GGML_BLAS ON CACHE BOOL "")
    set(GGML_BLAS_VENDOR "Generic" CACHE STRING "")
elseif(BACKEND STREQUAL "blis")
    set(GGML_BLAS ON CACHE BOOL "")
    set(GGML_BLAS_VENDOR "FLAME" CACHE STRING "")
elseif(NOT BACKEND STREQUAL "cpu")
    message(FATAL_ERROR "Unknown BACKEND '${BACKEND}' (expected cpu, blas, or blis)")
endif()
message(STATUS "MLqualcosa backend: ${BACKEND}  (GGML_BLAS=${GGML_BLAS}, GGML_BLAS_VENDOR=${GGML_BLAS_VENDOR})")

# Keep the llamacpp build scoped to what we actually need (llama-bench),
# same as before -- server/tests/examples stay off.
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "")
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "")
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "")
set(LLAMA_BUILD_COMMON ON CACHE BOOL "")
set(LLAMA_BUILD_TOOLS ON CACHE BOOL "")

# ONE build of llamacpp: this compiles ggml (used by mlops/ below) as part
# of building llama/common/llama-bench (used by inference/). Building
# mlops's own executables against this same `ggml` target, instead of each
# subfolder configuring its own separate ggml build, means ggml's CPU
# kernels only get compiled once for the whole project.
add_subdirectory(llamacpp)

# mlops/ just adds executables that link against the `ggml` target that
# llamacpp's own CMakeLists.txt already defined above -- no add_subdirectory
# of ggml here, and no separate project()/build tree.
add_subdirectory(mlops)
