#pragma once
#include <sycl/sycl.hpp>
#include "pstl/utils/elem_t.h"
#include "pstl/utils/sycl_utils.h"

namespace benchmark_inclusive_scan
{
    const auto inclusive_scan_sycl = [](auto && policy, const auto & input, auto & output) {
        auto & q       = pstl::sycl_utils::get_queue();
        const size_t n = input.size();

        sycl::buffer<pstl::elem_t> in_buf(input.data(),  sycl::range<1>(n));
        sycl::buffer<pstl::elem_t> out_buf(output.data(), sycl::range<1>(n));

        // Two-pass work-efficient inclusive scan (Blelloch)
        // Pass 1: upsweep (reduce)
        size_t padded = 1;
        while (padded < n) padded <<= 1;

        sycl::buffer<pstl::elem_t> tmp_buf(padded);

        // Copy input to tmp
        q.submit([&](sycl::handler & h) {
            auto in  = in_buf.template get_access<sycl::access::mode::read>(h);
            auto tmp = tmp_buf.template get_access<sycl::access::mode::write>(h);
            h.parallel_for(sycl::range<1>(padded), [=](sycl::id<1> i) {
                tmp[i] = (i < n) ? in[i] : 0;
            });
        }).wait();

        // Upsweep
        for (size_t stride = 1; stride < padded; stride <<= 1) {
            q.submit([&](sycl::handler & h) {
                auto tmp = tmp_buf.template get_access<sycl::access::mode::read_write>(h);
                size_t s = stride;
                size_t p = padded;
                h.parallel_for(sycl::range<1>(p / (2 * s)), [=](sycl::id<1> i) {
                    size_t idx = (i + 1) * 2 * s - 1;
                    if (idx < p) tmp[idx] += tmp[idx - s];
                });
            }).wait();
        }

        // Set last to 0 for exclusive scan base
        q.submit([&](sycl::handler & h) {
            auto tmp = tmp_buf.template get_access<sycl::access::mode::write>(h);
            size_t p = padded;
            h.single_task([=]() { tmp[p - 1] = 0; });
        }).wait();

        // Downsweep
        for (size_t stride = padded >> 1; stride >= 1; stride >>= 1) {
            q.submit([&](sycl::handler & h) {
                auto tmp = tmp_buf.template get_access<sycl::access::mode::read_write>(h);
                size_t s = stride;
                size_t p = padded;
                h.parallel_for(sycl::range<1>(p / (2 * s)), [=](sycl::id<1> i) {
                    size_t idx = (i + 1) * 2 * s - 1;
                    if (idx < p) {
                        pstl::elem_t t = tmp[idx - s];
                        tmp[idx - s]   = tmp[idx];
                        tmp[idx]      += t;
                    }
                });
            }).wait();
        }

        // Convert exclusive to inclusive and write output
        q.submit([&](sycl::handler & h) {
            auto in   = in_buf.template get_access<sycl::access::mode::read>(h);
            auto tmp  = tmp_buf.template get_access<sycl::access::mode::read>(h);
            auto out  = out_buf.template get_access<sycl::access::mode::write>(h);
            h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
                out[i] = tmp[i] + in[i];
            });
        }).wait();
    };
} // namespace benchmark_inclusive_scan
