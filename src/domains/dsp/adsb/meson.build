deps = [
    'JETSTREAM_LOADER_LIBMODES_AVAILABLE',
    'JETSTREAM_LOADER_ZLIB_AVAILABLE',
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and cfg_lst.get(x_dep, false)
endforeach

if all_deps_found
    src_lst += files([
        'block_impl.cc',
        'module_impl.cc',
        'module_impl_native_cpu.cc',
        'module_benchmarks.cc',
    ])

    module_test_lst += files(['module_tests.cc', 'block_tests.cc'])
endif
