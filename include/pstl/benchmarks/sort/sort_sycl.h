#pragma once
#include <sycl/sycl.hpp>
#include <unordered_map>
#include "pstl/utils/elem_t.h"
#include "pstl/utils/sycl_utils.h"
namespace benchmark_sort {
    namespace usm_cache {
        static std::unordered_map<size_t, pstl::elem_t*> d_in;
    }
    const auto sort_sycl = [](auto && policy, auto & input) {
        auto & q = pstl::sycl_utils::get_queue();
        const size_t n = input.size();
        if (usm_cache::d_in.find(n) == usm_cache::d_in.end()) {
            usm_cache::d_in[n] = sycl::malloc_device<pstl::elem_t>(n, q);
        }
        pstl::elem_t* d = usm_cache::d_in[n];
        q.memcpy(d, input.data(), n * sizeof(pstl::elem_t)).wait();
        for (size_t k=2; k<=n; k<<=1) {
            for (size_t j=k>>1; j>=1; j>>=1) {
                q.submit([&](sycl::handler & h) {
                    size_t kk=k, jj=j;
                    h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> tid) {
                        size_t i=tid[0], l=i^jj;
                        if (l>i) {
                            bool asc=((i&kk)==0);
                            if ((d[i]>d[l])==asc) {
                                pstl::elem_t t=d[i]; d[i]=d[l]; d[l]=t;
                            }
                        }
                    });
                }).wait();
            }
        }
        q.memcpy(input.data(), d, n * sizeof(pstl::elem_t)).wait();
    };
} // namespace benchmark_sort
