blueprint_gain_sources = files([
    'block_impl.cc',
    'module_benchmarks.cc',
    'module_impl.cc',
])

if plugin_device_cpu_enabled
    blueprint_gain_sources += files(['module_impl_native_cpu.cc'])
endif

plugin_sources += blueprint_gain_sources

plugin_test_lst += files([
    'block_tests.cc',
    'module_tests.cc',
])
