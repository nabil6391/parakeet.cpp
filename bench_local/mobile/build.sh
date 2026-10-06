#!/bin/bash
# build.sh ios-sim | ios | android  -> bench_local/mobile/build-<target>/bench
set -e
cd "$(dirname "$0")"
export PATH="$HOME/StudioProjects/parakeet-redux-investigation/.venv/bin:$PATH"
NDK=$HOME/Library/Android/sdk/ndk/29.0.14206865
case "$1" in
  ios-sim) A=(-DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT=iphonesimulator -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=16.0 -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16) ;;
  ios)     A=(-DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT=iphoneos -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=16.0 -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16) ;;
  # armv8.2-a+dotprod: the TQ2_0 sdot kernels; all phones since ~2018 have it
  android) A=(-DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28
              -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16) ;;
  *) echo "usage: $0 ios-sim|ios|android"; exit 1 ;;
esac
cmake -S . -B build-$1 -DCMAKE_BUILD_TYPE=Release "${A[@]}" >/dev/null
cmake --build build-$1 --target bench -j 8 2>&1 | grep -E "error|warning: unused|Linking|Built target bench" || true
ls -la build-$1/bench
