if not jst_is_browser and cfg_lst.get('JETSTREAM_LOADER_ZEROMQ_AVAILABLE', false)
    cfg_lst.set('JETSTREAM_MODULE_ZEROMQ_RECEIVER_AVAILABLE', true)
    cfg_lst.set('JETSTREAM_MODULE_ZEROMQ_RECEIVER_CPU_AVAILABLE', true)

    src_lst += files([
        'block_impl.cc',
        'module_impl.cc',
        'library/link.cc',
        'module_impl_native_cpu.cc',
    ])

    sum_lst += {'ZeroMQ Receiver': ['CPU']}
endif
