add_compile_definitions(PSTL_BENCH_USE_SYCL)
add_compile_definitions(PSTL_BENCH_BACKEND="SYCL")
add_compile_options(-fsycl)
add_link_options(-fsycl)
