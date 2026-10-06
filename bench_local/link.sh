#!/bin/bash
# link.sh <build-dir> <out-binary> : (re)links the bench harness against a build's static libparakeet + its ggml dylibs
cd "$(dirname "$0")/.."
clang++ -std=c++17 -O2 -I src -I include bench_local/bench.cpp -o "$2" "$1/libparakeet.a" \
  $(find "$1" -name "libggml*.dylib" | tr '\n' ' ') -framework Accelerate -framework Foundation || exit 1
echo "$(find "$1" -name "libggml*.dylib" -exec dirname {} \; | sort -u | tr '\n' ':')" > "$2.libpath"
