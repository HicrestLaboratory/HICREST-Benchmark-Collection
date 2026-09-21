backend="cpu"
blas_lib=""
blas_include=""
extra_args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)      backend="$2";      shift 2 ;;
        --blas-lib)     blas_lib="$2";     shift 2 ;;
        --blas-include) blas_include="$2"; shift 2 ;;
        --)             shift; extra_args=("$@"); break ;;
        *) echo "error: unknown argument '$1'" >&2; exit 1 ;;
    esac
done

cmake_args=(-DCMAKE_BUILD_TYPE=Release -DBACKEND="$backend")
[[ -n "$blas_lib"     ]] && cmake_args+=(-DBLAS_LIBRARIES="$blas_lib")
[[ -n "$blas_include" ]] && cmake_args+=(-DBLAS_INCLUDE_DIRS="$blas_include")

rm -rf build
mkdir build && cd build
cmake "${cmake_args[@]}" "${extra_args[@]}" ..
cmake --build . -j"$(nproc)"
cd ..