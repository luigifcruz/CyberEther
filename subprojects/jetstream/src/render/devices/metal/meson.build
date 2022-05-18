deps = [
    'JETSTREAM_BACKEND_METAL_AVAILABLE',
    'JETSTREAM_LOADER_GLFW_AVAILABLE',
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and cfg_lst.get(x_dep, false)
endforeach

if all_deps_found
    cfg_lst.set('JETSTREAM_RENDER_METAL_AVAILABLE', true)
    src_lst += files([
    ])
endif

sum_lst += {'Metal': all_deps_found}

