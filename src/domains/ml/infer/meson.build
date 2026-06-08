if cfg_lst.get('JETSTREAM_LOADER_ONNXRUNTIME_AVAILABLE', false)
    cfg_lst.set('JETSTREAM_MODULE_INFER_AVAILABLE', true)
    cfg_lst.set('JETSTREAM_MODULE_INFER_CPU_AVAILABLE', true)

    src_lst += files([
        'block_impl.cc',
        'module_impl.cc',
        'module_impl_native_cpu.cc',
    ])

    if cfg_lst.get('JETSTREAM_LOADER_ONNXRUNTIME_COREML_AVAILABLE', false)
        src_lst += files(['module_impl_metal.cc'])
    endif

    sum_lst += {'Infer': ['CPU']}
endif
