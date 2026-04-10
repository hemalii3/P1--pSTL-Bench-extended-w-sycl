# What Added to existing pSTL-Bench 

 file describes everything added to  existing pSTL-Bench repository for P1. The base repo is: https://github.com/parlab-tuwien/pSTL-Bench

---

## 1. SYCL Backend

5 new algorithm implementations using Intel DPC++ (icpx) and SYCL nd_range kernels.
files are in `include/pstl/benchmarks/<algorithm>/`.

### `include/pstl/benchmarks/find/find_sycl.h`
explicit nd_range kernel. work-item checks one element and uses
`sycl::atomic_ref::fetch_min` to record the earliest match index.
w-group size is passed directly as the nd_range local size.

### `include/pstl/benchmarks/for_each/for_each_sycl.h`
explicit nd_range kernel. Each work-item applies the kernel function to one
element. 
w-group size is passed directly as the nd_range local size.

### `include/pstl/benchmarks/reduce/reduce_sycl.h`
explicit nd_range kernel with local memory reduction.  work-group reduces
 chunk into local memory, then the partial results combined.
w-group size is passed directly as the nd_range local size.

### `include/pstl/benchmarks/inclusive_scan/inclusive_scan_sycl.h`
delgates to `oneapi::dpl::inclusive_scan` with a SYCL execution policy. oneDPL uses own internal config. compile-time wg_size parameter is not used by oneDPL here. goalis to write it better for gpu
### `include/pstl/benchmarks/sort/sort_sycl.h`
delegates to `oneapi::dpl::sort` with a SYCL execution policy.
oneDPL uses own internal radix sort. The compile-time wg_size parameter
is not used. confirmed w binary inspection showing oneDPL hardcodes its own
value. goal is same as inclusive scan.

---

## 2. Shared SYCL Utilities

### `include/pstl/utils/sycl_utils.h` (new file)
3 utilities shared across all SYCL algorithm implementations:

```cpp
// Wg size - set at compile time via -DPSTL_BENCH_SYCL_WG_SIZE=N
// Default is 256
constexpr size_t wg_size = PSTL_BENCH_SYCL_WG_SIZE;

// Rounds up global size to be a multiple of wg_size
// Required for nd_range kernels
inline size_t round_up_global_size(size_t n);

// Shared SYCL queue using default_selector_v
// Single queue instance reused across all kernels
inline sycl::queue & get_queue();
```

---

## 3. CMake Backend File

### `cmake/backends/SYCL.cmake` (new)
enables SYCL compilation with Intel DPC++:
- Sets `PSTL_BENCH_USE_SYCL` and `PSTL_BENCH_BACKEND="SYCL"` compile definitions
- Adds `-fsycl` compile and link flags

### `CMakeLists.txt` modified
Added work-group size parameter:
```cmake
if (NOT DEFINED PSTL_BENCH_SYCL_WG_SIZE)
    set(PSTL_BENCH_SYCL_WG_SIZE 256 CACHE STRING "SYCL work-group size")
endif()
add_compile_definitions(PSTL_BENCH_SYCL_WG_SIZE=${PSTL_BENCH_SYCL_WG_SIZE})
```
passes the wg_size value from CMake configure time through to the compiler
preprocessor, where `sycl_utils.h` picks it up as a compile-time constant.

---

## 4. build and run Scripts

### `build_baselines.sh` (new)
builds TBB, GNU, and HPX backends into `~/pstl-builds/` outside the repo.

### `build_all_wg.sh` (new)
builds 6 SYCL variants with different work-group sizes:
```bash
for WG in 32 64 128 256 512 1024; do
    cmake -DPSTL_BENCH_BACKEND=SYCL \
          -DCMAKE_CXX_COMPILER=icpx \
          -DPSTL_BENCH_SYCL_WG_SIZE=$WG \
          -S . -B ~/pstl-builds/sycl-wg${WG}
done
```
each produces a separate binary with a different compile-time wg_size.
Verified via CMakeCache and md5sum that all 6 binaries are genuinely different.

### `run_baselines.sh` (new)
runs TBB, GNU, HPX benchmarks with:
- Filter: `std::(find|for_each|inclusive_scan|reduce|sort)/` (
- 10 repetitions per benchmark
- 1s minimum time per benchmark
- Output: `results/tbb.json`, `results/gnu.json`, `results/hpx.json`

### `run_all_wg.sh` (new)
runs all 6 SYCL wg_size variants with the same settings.
Output: `results/sycl_wg32.json` through `results/sycl_wg1024.json`

### `reproduce.sh` (new)
 script: build → run → analyse. Supports `--skip-build` and
`--skip-run` flags to skip phases when binaries or results already exist.

---

## 5. Analysis and Plot Scripts

### `plot_results_stats.py` (new)
main plot script for the project. Loads all 9 JSON files (TBB, GNU, HPX,
SYCL wg32-1024), computes median and IQR from the 10 repetitions per benchmark,
and generates one plot per algorithm with error bands.
op: `plots_stats/` - 5 PNG files + `summary_stats.csv`

warns when coefficient of variation exceeds 5%

### `compare_wg_sizes.py` (new)
wg_size analysis script. Loads the 6 SYCL JSON files and produces:
- Overlay plots: all 6 wg_size variants on one chart per algorithm
- Heatmaps: pairwise mean relative difference between wg_size variants
  false positives at small N due to kernel launch overhead noise

op: `plots_wg_compare/` - 10 PNG files + 2 CSV files

### `run_analysis.sh` (new)
run `plot_results_stats.py` and `compare_wg_sizes.py` in sequence.

---

## 6. Results

`results/` directory contains JSON benchmark output for all 9 backends:
- `tbb.json`, `gnu.json`, `hpx.json` - baseline backends
- `sycl_wg32.json` through `sycl_wg1024.json` - SYCL variants

All results use 10 repetitions. Benchmark names follow  format:
`GNU-TBB/std::find/double/<size>/manual_time` (baseline)
`IntelLLVM-SYCL/sycl::find/double/<size>/manual_time` (SYCL)

---


# pSTL-Bench

pSTL-Bench is a benchmark suite designed to assist developers in evaluating the most suitable parallel
STL (Standard Template Library) backend for their needs.
This tool allows developers to benchmark a wide variety of parallel primitives and offers the flexibility to choose the
desired backend for execution during compile time.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)

## Introduction

pSTL-Bench is a resource for developers seeking to assess the performance and suitability of different
parallel STL backends.
By providing a rich benchmark suite, it facilitates the evaluation of parallel primitives across various
implementations, aiding in the selection of the optimal backend for specific requirements.

## Features

- Comprehensive benchmark suite for parallel STL backends
- Benchmarks a wide variety of parallel primitives
- Flexibility to choose the desired backend at compile time
- Facilitates performance comparison and evaluation of different implementations

## Getting Started

To run pSTL-Bench, follow these steps:

1. Clone the repository:

```shell
git clone https://github.com/parlab-tuwien/pSTL-Bench.git
```

2. Build the project with the desired parallel STL Backend

```shell
cmake -DCMAKE_BUILD_TYPE=Release -DPSTL_BENCH_BACKEND=TBB -DCMAKE_CXX_COMPILER=g++ -S . -B ./cmake-build-gcc
cmake --build cmake-build-gcc/ --target pSTL-Bench
```

One must define which backend to be used and which compiler.
You can define the backend with `-DPSTL_BENCH_BACKEND=...` and the compiler with `-DCMAKE_CXX_COMPILER=...`.
In the example above we will use g++ with TBB.
A list of supported backends can be seen in `./cmake/`.

Other options are:

* `-DPSTL_BENCH_DATA_TYPE=...` to define the data type (`int`, `float`, `double`...).
* `-DPSTL_BENCH_MIN_INPUT_SIZE=...` and `-DPSTL_BENCH_MAX_INPUT_SIZE=...` to define the range of input sizes.
* `-DPSTL_BENCH_USE_PAR_ALLOC=ON|OFF` to use a parallel allocator designed for NUMA systems.
* `-DPSTL_BENCH_USE_LIKWID=ON|OFF` and `-DPSTL_BENCH_USE_PAPI=ON|OFF` to use performance counters
  with [LIKWID](https://github.com/RRZE-HPC/likwid) or [PAPI](https://github.com/icl-utk-edu/papi).
* `-DPSTL_BENCH_GPU_CONTINUOUS_TRANSFERS=ON|OFF` to enable continuous transfers between the CPU and GPU so will be
  transferred between host and device before and after each kernel. When OFF, data will be transferred only once before
  the first call.

_Note_: we recommend to use `ccmake` to see all the possible flags and options.

## USAGE

After building the binary for a desired backend compiler pairing, you can simply call it.
Since we are using [Google benchmark](https://github.com/google/benchmark) under the hood, you can use all the possible
command line parameters.
For example:

```shell
./build/pSTL-Bench --benchmark_filter="std::sort"
```

The full set of options can be printed with `./pSTL-Bench --help`.

To get the full list of benchmarks, you can use the `--benchmark_list_tests` flag.

By default, `pSTL-Bench` will capture the `OMP_NUM_THREADS` environment variable to set the number of threads.
However, for [HPX](https://github.com/STEllAR-GROUP/hpx) argument `--hpx:threads` must be used.

Other environment variables that can be used are:

* `PSTL_BENCH_ABS_TOL` and `PSTL_BENCH_REL_TOL` to define the absolute and relative tolerance when asserting the results
  of floating point operations.

## Citation

If you use pSTL-Bench in your research, please cite the following papers:

```bibtex
@inproceedings{
  pstlbench-icpp24,
  title={Exploring Scalability in {C++} Parallel {STL} Implementations},
  author={Ruben Laso and Diego Krupitza and Sascha Hunold},
  booktitle={Proceedings of the 2024 International Conference on Parallel Processing},
  year={2024},
  doi={10.1145/3673038.3673065}
}

@misc{
  pstlbench2024,
  title={{pSTL-Bench}: A Micro-Benchmark Suite for Assessing Scalability of {C++} Parallel {STL} Implementations},
  author={Ruben Laso and Diego Krupitza and Sascha Hunold},
  year={2024},
  eprint={2402.06384},
  archivePrefix={arXiv},
  primaryClass={cs.DC}
}
```

## Dependencies

Some parallel STL backends have dependencies:

- TBB can be found on their [GitHub](https://github.com/oneapi-src/oneTBB) or
  their [website](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onetbb.html).
- HPX can be found on their [GitHub](https://github.com/STEllAR-GROUP/hpx) or
  their [website](https://hpx.stellar-group.org/).
- NVHPC can be found on their [website](https://developer.nvidia.com/hpc-sdk).
