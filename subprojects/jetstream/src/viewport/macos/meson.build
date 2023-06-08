deps = [
    'JETSTREAM_BACKEND_METAL_AVAILABLE',
    'JETSTREAM_LOADER_GLFW_AVAILABLE',
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and cfg_lst.get(x_dep, false)
endforeach

if host_machine.system() != 'darwin' 
    all_deps_found = false
endif

if all_deps_found
    subdir('tools')

    cfg_lst.set('JETSTREAM_VIEWPORT_MACOS_AVAILABLE', true)
    src_lst += files([
        'base.cc',
    ])
endif

sum_lst += {'macOS': all_deps_found}
