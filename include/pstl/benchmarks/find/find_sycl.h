#pragma once
#include <sycl/sycl.hpp>
#include "pstl/utils/elem_t.h"
#include "pstl/utils/sycl_utils.h"
namespace benchmark_find
{
    const auto find_sycl = [](auto && policy, const auto & input, const pstl::elem_t & target) {
        auto & q             = pstl::sycl_utils::get_queue();
        const size_t n       = input.size();
        const size_t global  = pstl::sycl_utils::round_up_global_size(n);
        const size_t wg      = pstl::sycl_utils::wg_size;
        size_t result = n;
        {
            sycl::buffer<pstl::elem_t> in_buf(input.data(), sycl::range<1>(n));
            sycl::buffer<size_t>       out_buf(&result, sycl::range<1>(1));
            q.submit([&](sycl::handler & h) {
                auto in  = in_buf.template get_access<sycl::access::mode::read>(h);
                auto out = out_buf.template get_access<sycl::access::mode::read_write>(h);
                h.parallel_for(sycl::nd_range<1>(global, wg), [=](sycl::nd_item<1> it) {
                    const size_t i = it.get_global_id(0);
                    if (i < n && in[i] == target) {
                        sycl::atomic_ref<size_t,
                            sycl::memory_order::relaxed,
                            sycl::memory_scope::device> atomic_out(out[0]);
                        atomic_out.fetch_min(i);
                    }
                });
            }).wait();
        } // buffer destructor flushes result back to host here
        return result == n ? input.end() : input.begin() + result;
    };
} // namespace benchmark_find
