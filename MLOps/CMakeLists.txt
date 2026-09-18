cmake_minimum_required(VERSION 3.14)
project(ggml_op_benchmarks CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Only build ggml (not the full llama.cpp lib/examples/server) — it is what
# actually implements GEMM/GEMV (ggml_mul_mat), GELU, and softmax on CPU.
# Because this is add_subdirectory'd from an outer project, GGML_STANDALONE
# is OFF inside ggml/CMakeLists.txt, so ggml's own tests/examples are not
# built as a side effect.
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

message(STATUS "ggml-bench backend: ${BACKEND}  (GGML_BLAS=${GGML_BLAS}, GGML_BLAS_VENDOR=${GGML_BLAS_VENDOR})")


add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/ggml ${CMAKE_CURRENT_BINARY_DIR}/ggml)

function(add_bench name)
    add_executable(${name} src/${name}.cpp)
    target_link_libraries(${name} PRIVATE ggml)
    target_include_directories(${name} PRIVATE src)
endfunction()

add_bench(gemm)
add_bench(gemv)
add_bench(gelu)
add_bench(softmax)
