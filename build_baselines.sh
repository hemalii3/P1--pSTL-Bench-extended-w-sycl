#!/bin/bash
source /opt/intel/oneapi/setvars.sh --force
BUILD_BASE=~/pstl-builds

for BACKEND in TBB GNU; do
    echo "Building ${BACKEND}..."
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DPSTL_BENCH_BACKEND=${BACKEND} \
          -DPSTL_BENCH_MAX_INPUT_SIZE=268435456 \
          -S . -B ${BUILD_BASE}/${BACKEND,,} > /dev/null 2>&1
    cmake --build ${BUILD_BASE}/${BACKEND,,} --target pSTL-Bench > /dev/null 2>&1
    echo "Done ${BACKEND}"
done
echo "Baseline builds complete."
