#!/bin/bash
# reproduce.sh — End-to-end reproducibility script for pSTL-Bench
# Usage:
#   bash reproduce.sh                  # full run: build + run + analyse
#   bash reproduce.sh --skip-build     # skip build (binaries already exist)
#   bash reproduce.sh --skip-run       # skip benchmarks (results already exist)
source /opt/intel/oneapi/setvars.sh --force
set -euo pipefail

SKIP_BUILD=false
SKIP_RUN=false

for arg in "$@"; do
    case $arg in
        --skip-build) SKIP_BUILD=true ;;
        --skip-run)   SKIP_RUN=true ;;
    esac
done

START_TIME=$(date +%s)
echo "========================================"
echo "  pSTL-Bench Reproduce Script"
echo "  $(date)"
echo "========================================"

# ── 0. oneAPI environment ─────────────────────────────────────────────────────
echo ""
echo "[0] Initialising Intel oneAPI environment..."
#source /opt/intel/oneapi/setvars.sh --force
echo "    icpx: $(icpx --version 2>&1 | head -1)"
echo "    g++:  $(g++ --version | head -1)"

# ── 1. Build ──────────────────────────────────────────────────────────────────
BUILD_BASE=~/pstl-builds

if [ "$SKIP_BUILD" = true ]; then
    echo ""
    echo "[1] Skipping build (--skip-build set)"
else
    echo ""
    echo "[1/3] Building TBB and GNU baselines..."
    for BACKEND in TBB GNU; do
        echo "    Building ${BACKEND}..."
        cmake -DCMAKE_BUILD_TYPE=Release \
              -DPSTL_BENCH_BACKEND=${BACKEND} \
              -DPSTL_BENCH_MAX_INPUT_SIZE=268435456 \
              -S . -B ${BUILD_BASE}/${BACKEND,,} > /dev/null 2>&1
        cmake --build ${BUILD_BASE}/${BACKEND,,} --target pSTL-Bench > /dev/null 2>&1
        echo "    Done ${BACKEND}"
    done

    echo ""
    echo "[1/3] Building SYCL wg_size variants (32 64 128 256 512 1024)..."
    for WG in 32 64 128 256 512 1024; do
        echo "    Building wg_size=$WG..."
        cmake -DCMAKE_BUILD_TYPE=Release \
              -DPSTL_BENCH_BACKEND=SYCL \
              -DCMAKE_CXX_COMPILER=icpx \
              -DPSTL_BENCH_MAX_INPUT_SIZE=268435456 \
              -DPSTL_BENCH_SYCL_WG_SIZE=$WG \
              -S . -B ${BUILD_BASE}/sycl-wg${WG} > /dev/null 2>&1
        cmake --build ${BUILD_BASE}/sycl-wg${WG} --target pSTL-Bench > /dev/null 2>&1
        echo "    Done wg_size=$WG"
    done
    echo "    All builds complete."
fi

# ── 2. Run benchmarks ─────────────────────────────────────────────────────────
FILTER="std::(find|for_each|inclusive_scan|reduce|sort)/"
mkdir -p results

if [ "$SKIP_RUN" = true ]; then
    echo ""
    echo "[2] Skipping benchmark runs (--skip-run set)"
else
    echo ""
    echo "[2/3] Running TBB and GNU baselines (10 reps each)..."
    for BACKEND in tbb gnu hpx; do
        echo "    Running ${BACKEND}..."
        ${BUILD_BASE}/${BACKEND}/pSTL-Bench \
            --benchmark_filter="${FILTER}" \
            --benchmark_repetitions=10 \
            --benchmark_min_time=1s \
            --benchmark_out=results/${BACKEND}.json \
            --benchmark_out_format=json
        echo "    Done ${BACKEND}"
    done

    echo ""
    echo "[2/3] Running SYCL wg_size variants (10 reps each)..."
    for WG in 32 64 128 256 512 1024; do
        echo "    Running wg_size=$WG..."
        ${BUILD_BASE}/sycl-wg${WG}/pSTL-Bench \
            --benchmark_filter="${FILTER}" \
            --benchmark_repetitions=10 \
            --benchmark_min_time=1s \
            --benchmark_out=results/sycl_wg${WG}.json \
            --benchmark_out_format=json
        echo "    Done wg_size=$WG"
    done
    echo "    All runs complete."
fi

# ── 3. Analysis ───────────────────────────────────────────────────────────────
echo ""
echo "[3/3] Running analysis pipeline..."
bash run_analysis.sh

# ── Summary ───────────────────────────────────────────────────────────────────
END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
echo ""
echo "========================================"
echo "  Done in $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"
echo "  Results:  ./results/"
echo "  Plots:    ./plots/  ./plots_stats/  ./plots_wg_compare/"
echo "========================================"
