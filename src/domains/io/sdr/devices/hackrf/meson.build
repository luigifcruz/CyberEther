src_lst += files([
    'sweep.cc',
    'sweep_decoder.cc',
    'transfer_queue.cc',
])

if cfg_lst.get('JETSTREAM_LOADER_HACKRF_AVAILABLE', false)
    src_lst += files('hackrf.cc')
endif

module_test_lst += files([
    'device_tests.cc',
])
