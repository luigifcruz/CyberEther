src_lst += files([
    'tools/quick_tune_budget.cc',
])

if cfg_lst.get('JETSTREAM_LOADER_BLADERF_AVAILABLE', false)
    src_lst += files([
        'enumerate.cc',
        'stream.cc',
        'sweep.cc',
    ])
endif

module_test_lst += files([
    'tests.cc',
])
