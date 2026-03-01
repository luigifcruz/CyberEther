deps = []

if jst_is_ios or jst_is_macos
    deps += 'JETSTREAM_LOADER_AUDIOTOOLBOX_AVAILABLE'
endif

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and cfg_lst.get(x_dep, false)
endforeach

if all_deps_found or (not jst_is_ios and not jst_is_macos)
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
        'block_impl.cc',
        'module_impl.cc',
        'module_impl_native_cpu.cc',
    ])

    sum_lst += {'Audio': ['CPU']}
endif

module_test_lst += files([
    'module_tests.cc',
    'block_tests.cc',
])
