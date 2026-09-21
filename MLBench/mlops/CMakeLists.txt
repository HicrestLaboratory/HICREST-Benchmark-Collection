# mlops/CMakeLists.txt
#
# No project()/add_subdirectory(ggml) here on purpose: the parent
# CMakeLists.txt already add_subdirectory(llamacpp)'d above this, which
# defines the `ggml` target. Linking against it here reuses those exact
# compiled objects instead of triggering a second ggml build.

function(add_bench name)
    add_executable(${name} src/${name}.cpp)
    target_link_libraries(${name} PRIVATE ggml)
    target_include_directories(${name} PRIVATE src)
endfunction()

add_bench(gemm)
add_bench(gemv)
add_bench(gelu)
add_bench(softmax)
