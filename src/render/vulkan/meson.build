deps = [
    'JETSTREAM_BACKEND_VULKAN_AVAILABLE',
    'JETSTREAM_SHADERS_VULKAN_AVAILABLE',
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and cfg_lst.get(x_dep, false)
endforeach

if all_deps_found
    subdir('tools')

    cfg_lst.set('JETSTREAM_RENDER_VULKAN_AVAILABLE', true)
    src_lst += files([
        'window.cc',
        'surface.cc',
        'program.cc',
        'kernel.cc',
        'texture.cc',
        'buffer.cc',
        'draw.cc',
        'vertex.cc',
    ])
endif

sum_lst += {'Vulkan': all_deps_found}
