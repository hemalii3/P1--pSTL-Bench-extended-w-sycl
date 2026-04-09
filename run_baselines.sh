#!/bin/bash
source /opt/intel/oneapi/setvars.sh --force
BUILD_BASE=~/pstl-builds
mkdir -p results

FILTER="std::(find|for_each|inclusive_scan|reduce|sort)/"

for BACKEND in tbb gnu hpx; do
    echo "Running ${BACKEND}..."
    ${BUILD_BASE}/${BACKEND}/pSTL-Bench \
        --benchmark_filter="${FILTER}" \
        --benchmark_repetitions=10 \
        --benchmark_min_time=1s \
        --benchmark_out=results/${BACKEND}.json \
        --benchmark_out_format=json
    echo "Done ${BACKEND}"
done
echo "Done. Results in ./results/"
