#pragma once
#include <sycl/sycl.hpp>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include "pstl/utils/elem_t.h"
#include "pstl/utils/sycl_utils.h"

namespace benchmark_sort
{
    const auto sort_sycl = [](auto && policy, auto & input) {
        auto & q = pstl::sycl_utils::get_queue();
        sycl::buffer<pstl::elem_t> buf(input.data(), sycl::range<1>(input.size()));
        auto dpl_policy = oneapi::dpl::execution::make_device_policy(q);
        oneapi::dpl::sort(dpl_policy, oneapi::dpl::begin(buf), oneapi::dpl::end(buf));
        q.wait();
    };
} // namespace benchmark_sort
