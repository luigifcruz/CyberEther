src_lst += files([
    'quick_tune_budget.cc',
])

if cfg_lst.get('JETSTREAM_LOADER_BLADERF_AVAILABLE', false)
    src_lst += files('bladerf.cc')
endif

module_test_lst += files([
    'device_tests.cc',
])
