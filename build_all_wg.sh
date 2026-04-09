#!/bin/bash
source /opt/intel/oneapi/setvars.sh --force
BUILD_BASE=~/pstl-builds

for WG in 32 64 128 256 512 1024; do
    echo "Building wg_size=$WG..."
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DPSTL_BENCH_BACKEND=SYCL \
          -DCMAKE_CXX_COMPILER=icpx \
          -DPSTL_BENCH_MAX_INPUT_SIZE=268435456 \
          -DPSTL_BENCH_SYCL_WG_SIZE=$WG \
          -S . -B ${BUILD_BASE}/sycl-wg${WG} > /dev/null 2>&1
    cmake --build ${BUILD_BASE}/sycl-wg${WG} --target pSTL-Bench > /dev/null 2>&1
    echo "Done wg_size=$WG"
done
echo "All builds complete."
