src_lst += files([
    'module_impl.cc',
    'common.cc',
    'lineplot.cc',
    'waterfall.cc',
    'waterfall_3d.cc',
    'module_impl_native_cpu.cc',
    'module_benchmarks.cc',
])

if cfg_lst.get('JETSTREAM_BACKEND_CUDA_AVAILABLE', false)
    src_lst += files(['module_impl_native_cuda.cc'])
endif

module_test_lst += files(['module_tests.cc'])
