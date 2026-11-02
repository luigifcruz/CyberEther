if not jst_is_browser and cfg_lst.get('JETSTREAM_LOADER_ZEROMQ_AVAILABLE', false)
    cfg_lst.set('JETSTREAM_MODULE_ZEROMQ_TRANSMITTER_AVAILABLE', true)
    cfg_lst.set('JETSTREAM_MODULE_ZEROMQ_TRANSMITTER_CPU_AVAILABLE', true)

    src_lst += files([
        'block_impl.cc',
        'module_impl.cc',
        'module_impl_native_cpu.cc',
    ])

    sum_lst += {'ZeroMQ Transmitter': ['CPU']}
endif
