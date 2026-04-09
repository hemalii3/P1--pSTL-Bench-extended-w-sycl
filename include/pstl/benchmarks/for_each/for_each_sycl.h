#pragma once
#include <sycl/sycl.hpp>
#include "pstl/utils/elem_t.h"
#include "pstl/utils/sycl_utils.h"

namespace benchmark_for_each
{
    const auto for_each_sycl = [](auto && policy, auto & input, auto kernel) {
        auto & q            = pstl::sycl_utils::get_queue();
        const size_t n      = input.size();
        const size_t global = pstl::sycl_utils::round_up_global_size(n);
        const size_t wg     = pstl::sycl_utils::wg_size;

        sycl::buffer<pstl::elem_t> buf(input.data(), sycl::range<1>(n));

        q.submit([&](sycl::handler & h) {
            auto acc = buf.template get_access<sycl::access::mode::read_write>(h);
            h.parallel_for(sycl::nd_range<1>(global, wg), [=](sycl::nd_item<1> it) {
                const size_t i = it.get_global_id(0);
                if (i < n) kernel(acc[i]);
            });
        }).wait();
    };
} // namespace benchmark_for_each
