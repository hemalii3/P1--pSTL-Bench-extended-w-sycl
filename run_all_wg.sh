#!/bin/bash
source /opt/intel/oneapi/setvars.sh --force
BUILD_BASE=~/pstl-builds
mkdir -p results

for WG in 32 64 128 256 512 1024; do
    echo "Running wg_size=$WG..."
    ${BUILD_BASE}/sycl-wg${WG}/pSTL-Bench \
        --benchmark_filter="IntelLLVM-SYCL" \
        --benchmark_repetitions=10 \
        --benchmark_min_time=1s \
        --benchmark_out=results/sycl_wg${WG}.json \
        --benchmark_out_format=json
    echo "Done wg_size=$WG"
done
echo "All runs complete. Results in ./results/"
