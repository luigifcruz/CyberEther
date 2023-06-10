deps = [
    'JETSTREAM_LOADER_GLFW_AVAILABLE',
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and cfg_lst.get(x_dep, false)
endforeach

if all_deps_found
    subdir('tools')
    
    cfg_lst.set('JETSTREAM_VIEWPORT_GLFW_AVAILABLE', true)

    if cfg_lst.get('JETSTREAM_BACKEND_METAL_AVAILABLE', false)
        src_lst += files([
            'metal.cc',
        ])
    endif
    
    if cfg_lst.get('JETSTREAM_BACKEND_VULKAN_AVAILABLE', false)
        src_lst += files([
            'vulkan.cc',
        ])
    endif
endif

sum_lst += {'GLFW': all_deps_found}