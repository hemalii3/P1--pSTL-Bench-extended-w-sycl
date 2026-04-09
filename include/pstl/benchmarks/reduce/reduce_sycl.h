#pragma once
#include <sycl/sycl.hpp>
#include "pstl/utils/elem_t.h"
#include "pstl/utils/sycl_utils.h"

namespace benchmark_reduce
{
    const auto reduce_sycl = [](auto && policy, const auto & input) {
        auto & q            = pstl::sycl_utils::get_queue();
        const size_t n      = input.size();
        const size_t global = pstl::sycl_utils::round_up_global_size(n);
        const size_t wg     = pstl::sycl_utils::wg_size;

        pstl::elem_t result = 0;
        {
            sycl::buffer<pstl::elem_t> in_buf(input.data(), sycl::range<1>(n));
            sycl::buffer<pstl::elem_t> out_buf(&result, sycl::range<1>(1));

            q.submit([&](sycl::handler & h) {
                auto in = in_buf.template get_access<sycl::access::mode::read>(h);
                h.parallel_for(
                    sycl::nd_range<1>(global, wg),
                    sycl::reduction(out_buf, h, pstl::elem_t{}, sycl::plus<pstl::elem_t>()),
                    [=](sycl::nd_item<1> it, auto & sum) {
                        const size_t i = it.get_global_id(0);
                        if (i < n) sum += in[i];
                    });
            }).wait();
        }

        return result;
    };
} // namespace benchmark_reduce
