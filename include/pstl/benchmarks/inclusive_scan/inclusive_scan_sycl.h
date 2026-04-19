#pragma once
#include <sycl/sycl.hpp>
#include <unordered_map>
#include "pstl/utils/elem_t.h"
#include "pstl/utils/sycl_utils.h"
namespace benchmark_inclusive_scan {
    namespace usm_cache {
        static std::unordered_map<size_t, pstl::elem_t*> d_in;
        static std::unordered_map<size_t, pstl::elem_t*> d_out;
        static std::unordered_map<size_t, pstl::elem_t*> d_tmp;
        static std::unordered_map<size_t, size_t> padded;
    }
    const auto inclusive_scan_sycl = [](auto && policy, const auto & input, auto & output) {
        auto & q = pstl::sycl_utils::get_queue();
        const size_t n = input.size();
        size_t p = 1;
        while (p < n) p <<= 1;
        if (usm_cache::d_in.find(n) == usm_cache::d_in.end()) {
            usm_cache::d_in[n]  = sycl::malloc_device<pstl::elem_t>(n, q);
            usm_cache::d_out[n] = sycl::malloc_device<pstl::elem_t>(n, q);
            usm_cache::d_tmp[n] = sycl::malloc_device<pstl::elem_t>(p, q);
            usm_cache::padded[n] = p;
            q.memcpy(usm_cache::d_in[n], input.data(), n * sizeof(pstl::elem_t)).wait();
        }
        pstl::elem_t* di = usm_cache::d_in[n];
        pstl::elem_t* dout = usm_cache::d_out[n];
        pstl::elem_t* dt = usm_cache::d_tmp[n];
        q.submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(p), [=](sycl::id<1> i) {
                dt[i] = (i < n) ? di[i] : 0;
            });
        }).wait();
        for (size_t s = 1; s < p; s <<= 1) {
            q.submit([&](sycl::handler & h) {
                size_t ss=s, pp=p;
                h.parallel_for(sycl::range<1>(pp/(2*ss)), [=](sycl::id<1> i) {
                    size_t idx=(i+1)*2*ss-1;
                    if (idx<pp) dt[idx]+=dt[idx-ss];
                });
            }).wait();
        }
        q.submit([&](sycl::handler & h) {
            size_t pp=p;
            h.single_task([=]() { dt[pp-1]=0; });
        }).wait();
        for (size_t s = p>>1; s >= 1; s >>= 1) {
            q.submit([&](sycl::handler & h) {
                size_t ss=s, pp=p;
                h.parallel_for(sycl::range<1>(pp/(2*ss)), [=](sycl::id<1> i) {
                    size_t idx=(i+1)*2*ss-1;
                    if (idx<pp) {
                        pstl::elem_t t=dt[idx-ss];
                        dt[idx-ss]=dt[idx];
                        dt[idx]+=t;
                    }
                });
            }).wait();
        }
        q.submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
                dout[i]=dt[i]+di[i];
            });
        }).wait();
        q.memcpy(output.data(), dout, n * sizeof(pstl::elem_t)).wait();
    };
} // namespace benchmark_inclusive_scan
