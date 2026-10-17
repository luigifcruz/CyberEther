src_lst += files([
    'tools/sweep_layout.cc',
])

if cfg_lst.get('JETSTREAM_LOADER_PLUTOSDR_AVAILABLE', false)
    src_lst += files([
        'enumerate.cc',
        'stream.cc',
        'sweep.cc',
    ])
endif

module_test_lst += files([
    'tests.cc',
])
