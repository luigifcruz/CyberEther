inference_opt = get_option('inference')

if not inference_opt.disabled()
    ort_dep = dependency(['libonnxruntime', 'onnxruntime'], required: inference_opt.enabled())
    if ort_dep.found()
        dep_lst += [ort_dep]
        cfg_lst.set('JETSTREAM_LOADER_ONNXRUNTIME_AVAILABLE', true)
        ldr_lst += {'ONNX Runtime': true}
    else
        ldr_lst += {'ONNX Runtime': false}
    endif
else
    ldr_lst += {'ONNX Runtime': false}
endif
