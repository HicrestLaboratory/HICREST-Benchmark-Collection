backend="cpu"
extra_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend) backend="$2"; shift 2 ;;
        --)        shift; extra_args=("$@"); break ;;
        *) echo "error: unknown argument '$1'" >&2; exit 1 ;;
    esac
done
rm -rf build && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBACKEND="$backend" "${extra_args[@]}" ..
cmake --build . -j"$(nproc)"
cd ..