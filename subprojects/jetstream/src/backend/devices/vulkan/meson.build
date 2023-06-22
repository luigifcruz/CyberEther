deps = [
    'JETSTREAM_LOADER_VULKAN_AVAILABLE',
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and cfg_lst.get(x_dep, false)
endforeach

if all_deps_found
    cfg_lst.set('JETSTREAM_BACKEND_VULKAN_AVAILABLE', true)
    src_lst += files([
        'base.cc',
    ])
endif

sum_lst += {'Vulkan': all_deps_found}
