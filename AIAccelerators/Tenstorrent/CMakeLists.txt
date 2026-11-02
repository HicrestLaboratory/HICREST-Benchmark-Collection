cmake_minimum_required(VERSION 3.10)
project(TTNN_ADDMM_EXAMPLE VERSION 1.0 LANGUAGES CXX)

option(BUILD_TENSTORRENT_BENCHMARKS "Build the Tenstorrent benchmarks" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(BUILD_TENSTORRENT_BENCHMARKS)
    set(ARGPARSE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../AI-accelerators-benchmark/tenstorrent/third_party/argparse/include" CACHE PATH "argparse include directory")
    if(NOT DEFINED TT_METAL_HOME)
        if(DEFINED ENV{TT_METAL_HOME})
            set(TT_METAL_HOME $ENV{TT_METAL_HOME})
        else()
            message(FATAL_ERROR "Set TT_METAL_HOME to the root of your tt-metal checkout.")
        endif()
    endif()
    if(NOT DEFINED TT_METAL_RUNTIME_ROOT)
        if(DEFINED ENV{TT_METAL_RUNTIME_ROOT})
            set(TT_METAL_RUNTIME_ROOT $ENV{TT_METAL_RUNTIME_ROOT})
        else()
            message(FATAL_ERROR "Set TT_METAL_RUNTIME_ROOT to the root of your tt-metal checkout.")
        endif()
    endif()

    add_executable(gemm src/gemm_gemv.cpp)
    add_executable(non_linearity src/non_linearity.cpp)
    target_compile_options(gemm PRIVATE -O3 -Wno-deprecated-declarations)
    target_compile_options(non_linearity PRIVATE -O3 -Wno-deprecated-declarations)


    add_subdirectory(${TT_METAL_HOME} ${CMAKE_BINARY_DIR}/tt-metal-build)
    set(TT_LIBRARIES Metalium::Metal TT::Metalium::HostDevCommon
        TT::Metalium::Common ttnncpp)
    target_link_libraries(gemm PRIVATE ${TT_LIBRARIES})
    target_link_libraries(non_linearity PRIVATE ${TT_LIBRARIES})

    set(TT_INCLUDE_DIRECTORIES
        ${TT_METAL_HOME}
        ${TT_METAL_HOME}/tt_metal/api
        ${TT_METAL_HOME}/tt_metal/hw/inc
        ${TT_METAL_HOME}/tt_metal/third_party/tracy/public
        ${TT_METAL_HOME}/ttnn
        ${TT_METAL_HOME}/ttnn/api
        ${TT_METAL_HOME}/ttnn/cpp
        ${ARGPARSE_INCLUDE_DIR})
    target_include_directories(gemm PRIVATE ${TT_INCLUDE_DIRECTORIES})
    target_include_directories(non_linearity PRIVATE ${TT_INCLUDE_DIRECTORIES})
endif()