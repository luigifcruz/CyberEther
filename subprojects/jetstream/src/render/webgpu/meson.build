deps = [
    'JETSTREAM_BACKEND_WEBGPU_AVAILABLE',
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and cfg_lst.get(x_dep, false)
endforeach

if all_deps_found
    subdir('tools')

    cfg_lst.set('JETSTREAM_RENDER_WEBGPU_AVAILABLE', true)
    src_lst += files([
        'window.cc',
        'surface.cc',
        'program.cc',
        'texture.cc',
        'buffer.cc',
        'draw.cc',
        'vertex.cc',
    ])
endif

sum_lst += {'WebGPU': all_deps_found}
