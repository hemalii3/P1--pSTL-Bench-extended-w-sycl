#pragma once

#include "pstl/utils/benchmark_naming.h"
#include "inclusive_scan_utils.h"
#include "inclusive_scan_std.h"

#ifdef PSTL_BENCH_USE_HPX
#include "inclusive_scan_hpx.h"
#endif

#ifdef PSTL_BENCH_USE_SYCL
#include "inclusive_scan_sycl.h"
#endif

//region inclusive_scan_std
template<class Policy>
static void inclusive_scan_std_wrapper(benchmark::State & state)
{
	benchmark_inclusive_scan::benchmark_wrapper<Policy>(state, benchmark_inclusive_scan::inclusive_scan_std);
}

#define INCLUSIVE_SCAN_SEQ_WRAPPER                                                    \
	BENCHMARK_TEMPLATE1(inclusive_scan_std_wrapper, std::execution::sequenced_policy) \
	    ->Name(PSTL_BENCH_BENCHMARK_NAME_WITH_BACKEND("SEQ", "std::inclusive_scan"))  \
	    ->PSTL_BENCH_BENCHMARK_PARAMETERS

#ifdef PSTL_BENCH_USE_PSTL
#define INCLUSIVE_SCAN_STD_WRAPPER                                                               \
	BENCHMARK_TEMPLATE1(inclusive_scan_std_wrapper, std::execution::parallel_unsequenced_policy) \
	    ->Name(PSTL_BENCH_BENCHMARK_NAME("std::inclusive_scan"))                                 \
	    ->PSTL_BENCH_BENCHMARK_PARAMETERS
#else
#define INCLUSIVE_SCAN_STD_WRAPPER
#endif
//endregion inclusive_scan_std

//region inclusive_scan_hpx
#ifdef PSTL_BENCH_USE_HPX
template<class Policy>
static void inclusive_scan_hpx_wrapper(benchmark::State & state)
{
	benchmark_inclusive_scan::benchmark_wrapper<Policy>(state, benchmark_inclusive_scan::inclusive_scan_hpx);
}
#define INCLUSIVE_SCAN_HPX_WRAPPER                                                               \
	BENCHMARK_TEMPLATE1(inclusive_scan_hpx_wrapper, hpx::execution::parallel_unsequenced_policy) \
	    ->Name(PSTL_BENCH_BENCHMARK_NAME("hpx::inclusive_scan"))                                 \
	    ->PSTL_BENCH_BENCHMARK_PARAMETERS
#else
#define INCLUSIVE_SCAN_HPX_WRAPPER
#endif
//endregion inclusive_scan_hpx

//region inclusive_scan_sycl
#ifdef PSTL_BENCH_USE_SYCL
template<class Policy>
static void inclusive_scan_sycl_wrapper(benchmark::State & state)
{
	benchmark_inclusive_scan::benchmark_wrapper<Policy>(state, benchmark_inclusive_scan::inclusive_scan_sycl);
}
#define INCLUSIVE_SCAN_SYCL_WRAPPER                                                               \
	BENCHMARK_TEMPLATE1(inclusive_scan_sycl_wrapper, std::execution::parallel_unsequenced_policy) \
	    ->Name(PSTL_BENCH_BENCHMARK_NAME("sycl::inclusive_scan"))                                 \
	    ->PSTL_BENCH_BENCHMARK_PARAMETERS
#else
#define INCLUSIVE_SCAN_SYCL_WRAPPER
#endif
//endregion inclusive_scan_sycl

#define INCLUSIVE_SCAN_GROUP     \
	INCLUSIVE_SCAN_SEQ_WRAPPER   \
	INCLUSIVE_SCAN_STD_WRAPPER   \
	INCLUSIVE_SCAN_HPX_WRAPPER   \
	INCLUSIVE_SCAN_SYCL_WRAPPER
INCLUSIVE_SCAN_GROUP
