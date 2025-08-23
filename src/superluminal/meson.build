deps = [
    'JETSTREAM_MODULE_FFT_AVAILABLE',
    'JETSTREAM_MODULE_INVERT_AVAILABLE',
    'JETSTREAM_MODULE_WINDOW_AVAILABLE',
    'JETSTREAM_MODULE_MULTIPLY_AVAILABLE',
    'JETSTREAM_MODULE_AMPLITUDE_AVAILABLE',
    'JETSTREAM_MODULE_SCALE_AVAILABLE',
    'JETSTREAM_MODULE_LINEPLOT_AVAILABLE',
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and cfg_lst.get(x_dep, false)
endforeach

if all_deps_found
    cfg_lst.set('JETSTREAM_SUPERLUMINAL_AVAILABLE', true)

    src_lst += files([
        'base.cc',
        'dmi_module.cc',
    ])
endif
