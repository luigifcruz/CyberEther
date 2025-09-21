deps = [
    'JETSTREAM_LOADER_GSTREAMER_AVAILABLE',
    'JETSTREAM_LOADER_CPPHTTPLIB_AVAILABLE',
    'JETSTREAM_LOADER_FTXUI_AVAILABLE',
    'JETSTREAM_LOADER_QRENCODE_AVAILABLE',
    'JETSTREAM_LOADER_JSON_AVAILABLE',
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and cfg_lst.get(x_dep, false)
endforeach

if all_deps_found
    cfg_lst.set('JETSTREAM_VIEWPORT_HEADLESS_AVAILABLE', true)

    src_lst += files([
        'remote.cc',
    ])

    if cfg_lst.get('JETSTREAM_BACKEND_VULKAN_AVAILABLE', false)
        src_lst += files([
            'vulkan.cc',
        ])
    endif
endif

sum_lst += {'Headless': all_deps_found}
