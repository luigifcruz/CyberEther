deps = [
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and cfg_lst.get(x_dep, false)
endforeach

if all_deps_found
    cfg_lst.set('JETSTREAM_VIEWPORT_HEADLESS_AVAILABLE', true)

    src_lst += files([
        'endpoint.cc',
    ])
    
    if cfg_lst.get('JETSTREAM_BACKEND_VULKAN_AVAILABLE', false)
        src_lst += files([
            'vulkan.cc',
        ])
    endif
endif

sum_lst += {'Headless': all_deps_found}