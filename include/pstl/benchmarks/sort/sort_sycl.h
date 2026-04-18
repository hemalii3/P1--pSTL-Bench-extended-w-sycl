#pragma once
#include <sycl/sycl.hpp>
#include "pstl/utils/elem_t.h"
#include "pstl/utils/sycl_utils.h"

namespace benchmark_sort
{
    const auto sort_sycl = [](auto && policy, auto & input) {
        auto & q      = pstl::sycl_utils::get_queue();
        const size_t n = input.size();

        sycl::buffer<pstl::elem_t> buf(input.data(), sycl::range<1>(n));

        // Bitonic sort — works correctly for power-of-two sizes
        // For non-power-of-two, elements beyond n are not touched
        for (size_t k = 2; k <= n; k <<= 1) {
            for (size_t j = k >> 1; j >= 1; j >>= 1) {
                q.submit([&](sycl::handler & h) {
                    auto data = buf.template get_access<sycl::access::mode::read_write>(h);
                    size_t kk = k, jj = j;
                    h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> tid) {
                        size_t i = tid[0];
                        size_t l = i ^ jj;
                        if (l > i) {
                            bool ascending = ((i & kk) == 0);
                            if ((data[i] > data[l]) == ascending) {
                                pstl::elem_t tmp = data[i];
                                data[i] = data[l];
                                data[l] = tmp;
                            }
                        }
                    });
                }).wait();
            }
        }
    };
} // namespace benchmark_sort
