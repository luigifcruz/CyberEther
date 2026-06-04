if jst_is_browser or cfg_lst.get('JETSTREAM_LOADER_CPPHTTPLIB_AVAILABLE', false)
    cfg_lst.set('JETSTREAM_MODULE_WEBSOCKET_AVAILABLE', true)
    cfg_lst.set('JETSTREAM_MODULE_WEBSOCKET_CPU_AVAILABLE', true)

    src_lst += files([
        'block_impl.cc',
        'module_impl.cc',
        'module_impl_native_cpu.cc',
    ])

    sum_lst += {'WebSocket': ['CPU']}
endif

module_test_lst += files([
    'module_tests.cc',
    'block_tests.cc',
])
