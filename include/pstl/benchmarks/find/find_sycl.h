#pragma once
#include <sycl/sycl.hpp>
#include <unordered_map>
#include "pstl/utils/elem_t.h"
#include "pstl/utils/sycl_utils.h"
namespace benchmark_find {
    namespace usm_cache {
        static std::unordered_map<size_t, pstl::elem_t*> d_in;
        static std::unordered_map<size_t, size_t*> d_res;
    }
    const auto find_sycl = [](auto && policy, const auto & input, const pstl::elem_t & target) {
        auto & q = pstl::sycl_utils::get_queue();
        const size_t n = input.size();
        const size_t global = pstl::sycl_utils::round_up_global_size(n);
        const size_t wg = pstl::sycl_utils::wg_size;
        if (usm_cache::d_in.find(n) == usm_cache::d_in.end()) {
            usm_cache::d_in[n]  = sycl::malloc_device<pstl::elem_t>(n, q);
            usm_cache::d_res[n] = sycl::malloc_device<size_t>(1, q);
            q.memcpy(usm_cache::d_in[n], input.data(), n * sizeof(pstl::elem_t)).wait();
        }
        pstl::elem_t* d = usm_cache::d_in[n];
        size_t* r = usm_cache::d_res[n];
        size_t hr = n;
        q.memcpy(r, &hr, sizeof(size_t)).wait();
        q.submit([&](sycl::handler & h) {
            h.parallel_for(sycl::nd_range<1>(global, wg), [=](sycl::nd_item<1> it) {
                const size_t i = it.get_global_id(0);
                if (i < n && d[i] == target) {
                    sycl::atomic_ref<size_t, sycl::memory_order::relaxed,
                        sycl::memory_scope::device> ar(*r);
                    ar.fetch_min(i);
                }
            });
        }).wait();
        q.memcpy(&hr, r, sizeof(size_t)).wait();
        return hr == n ? input.end() : input.begin() + hr;
    };
} // namespace benchmark_find
