deps = [
    'JETSTREAM_BACKEND_VULKAN_AVAILABLE',
    'JETSTREAM_LOADER_GLFW_AVAILABLE',
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and cfg_lst.get(x_dep, false)
endforeach

if all_deps_found
    subdir('tools')

    cfg_lst.set('JETSTREAM_VIEWPORT_LINUX_AVAILABLE', true)
    src_lst += files([
        'base.cc',
    ])
endif

sum_lst += {'Linux': all_deps_found}
