src_lst += files([
    'tools/firmware.cc',
    'tools/sweep_decoder.cc',
    'tools/transfer_queue.cc',
])

if cfg_lst.get('JETSTREAM_LOADER_HACKRF_AVAILABLE', false)
    src_lst += files([
        'enumerate.cc',
        'stream.cc',
        'sweep.cc',
    ])
endif

module_test_lst += files([
    'tests.cc',
])
