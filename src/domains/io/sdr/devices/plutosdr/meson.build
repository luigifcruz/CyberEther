src_lst += files([
    'sweep_buffer.cc',
])

if cfg_lst.get('JETSTREAM_LOADER_PLUTOSDR_AVAILABLE', false)
    src_lst += files('plutosdr.cc')
endif

module_test_lst += files([
    'device_tests.cc',
])
