src_lst += files([
    'block_impl.cc',
    'module_impl.cc',
    'module_impl_native_cpu.cc',
])

if cfg_lst.get('JETSTREAM_BACKEND_CUDA_AVAILABLE', false)
    src_lst += files(['module_impl_native_cuda.cc'])
endif

module_test_lst += files(['module_tests.cc', 'block_tests.cc'])
