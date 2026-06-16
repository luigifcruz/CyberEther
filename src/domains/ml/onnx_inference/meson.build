if cfg_lst.get('JETSTREAM_LOADER_ONNXRUNTIME_AVAILABLE', false)
    cfg_lst.set('JETSTREAM_MODULE_ONNX_INFERENCE_AVAILABLE', true)
    cfg_lst.set('JETSTREAM_MODULE_ONNX_INFERENCE_CPU_AVAILABLE', true)

    src_lst += files([
        'block_impl.cc',
        'module_impl.cc',
        'module_impl_native_cpu.cc',
    ])

    sum_lst += {'ONNX Inference': ['CPU']}
endif
