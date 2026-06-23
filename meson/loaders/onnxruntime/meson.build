inference_opt = get_option('inference')

if inference_opt.enabled()
    dep_lst += [
        dependency(['libonnxruntime', 'onnxruntime'], required: true)
    ]
    cfg_lst.set('JETSTREAM_LOADER_ONNXRUNTIME_AVAILABLE', true)
    ldr_lst += {'ONNX Runtime': true}
else
    ldr_lst += {'ONNX Runtime': false}
endif
