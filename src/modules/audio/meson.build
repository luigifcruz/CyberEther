deps = [
]

if jst_is_ios or jst_is_macos
    deps += 'JETSTREAM_LOADER_AUDIOTOOLBOX_AVAILABLE'
endif

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and cfg_lst.get(x_dep, false)
endforeach

if all_deps_found
    cfg_lst.set('JETSTREAM_MODULE_AUDIO_AVAILABLE', true)
    cfg_lst.set('JETSTREAM_MODULE_AUDIO_CPU_AVAILABLE', true)

    if jst_is_ios or jst_is_macos
        src_lst += files([
            'miniaudio.mm',
        ])
    else
        src_lst += files([
            'miniaudio.cc',
        ])
    endif

    src_lst += files([
        'generic.cc',
    ])

    sum_lst += {'Audio': ['CPU']}
endif