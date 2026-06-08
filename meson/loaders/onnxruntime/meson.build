inference_opt = get_option('inference')

if not inference_opt.disabled()
    # Homebrew installs as 'libonnxruntime'; system packages as 'onnxruntime'.
    ort_dep = dependency('libonnxruntime', required: false)
    if not ort_dep.found()
        ort_dep = dependency('onnxruntime', required: false)
    endif

    if inference_opt.enabled() and not ort_dep.found()
        error('inference=enabled but ONNX Runtime was not found. ' +
              'Install with: brew install onnxruntime  (macOS) or ' +
              'pip install onnxruntime  (Linux/Windows).')
    endif

    if ort_dep.found()
        cfg_lst.set('JETSTREAM_LOADER_ONNXRUNTIME_AVAILABLE', true)
        dep_lst += [ort_dep]

        # CoreML EP ships with the Homebrew ORT build on macOS.
        if jst_is_macos
            cfg_lst.set('JETSTREAM_LOADER_ONNXRUNTIME_COREML_AVAILABLE', true)
        endif
    endif

    ldr_lst += {'ONNX Runtime': ort_dep.found()}
else
    ldr_lst += {'ONNX Runtime': false}
endif
