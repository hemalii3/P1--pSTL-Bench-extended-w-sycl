#pragma once
#include <sycl/sycl.hpp>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/numeric>
#include "pstl/utils/elem_t.h"
#include "pstl/utils/sycl_utils.h"

namespace benchmark_inclusive_scan
{
    const auto inclusive_scan_sycl = [](auto && policy, const auto & input, auto & output) {
        auto & q       = pstl::sycl_utils::get_queue();
        const size_t n = input.size();

        sycl::buffer<pstl::elem_t> in_buf(input.data(),  sycl::range<1>(n));
        sycl::buffer<pstl::elem_t> out_buf(output.data(), sycl::range<1>(n));

        auto dpl_policy = oneapi::dpl::execution::make_device_policy(q);
        oneapi::dpl::inclusive_scan(dpl_policy,
            oneapi::dpl::begin(in_buf), oneapi::dpl::end(in_buf),
            oneapi::dpl::begin(out_buf));
        q.wait();
    };
} // namespace benchmark_inclusive_scan
