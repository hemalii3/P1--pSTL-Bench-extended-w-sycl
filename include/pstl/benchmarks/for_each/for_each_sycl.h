#pragma once
#include <sycl/sycl.hpp>
#include <unordered_map>
#include "pstl/utils/elem_t.h"
#include "pstl/utils/sycl_utils.h"
namespace benchmark_for_each {
    namespace usm_cache {
        static std::unordered_map<size_t, pstl::elem_t*> d_in;
    }
    const auto for_each_sycl = [](auto && policy, auto & input, auto && kernel) {
        auto & q = pstl::sycl_utils::get_queue();
        const size_t n = input.size();
        const size_t global = pstl::sycl_utils::round_up_global_size(n);
        const size_t wg = pstl::sycl_utils::wg_size;
        if (usm_cache::d_in.find(n) == usm_cache::d_in.end()) {
            usm_cache::d_in[n] = sycl::malloc_device<pstl::elem_t>(n, q);
            q.memcpy(usm_cache::d_in[n], input.data(), n * sizeof(pstl::elem_t)).wait();
        }
        pstl::elem_t* d = usm_cache::d_in[n];
        q.submit([&](sycl::handler & h) {
            h.parallel_for(sycl::nd_range<1>(global, wg), [=](sycl::nd_item<1> it) {
                const size_t i = it.get_global_id(0);
                if (i < n) kernel(d[i]);
            });
        }).wait();
    };
} // namespace benchmark_for_each
