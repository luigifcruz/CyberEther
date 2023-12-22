deps = [
    'JETSTREAM_BACKEND_METAL_AVAILABLE',
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and cfg_lst.get(x_dep, false)
endforeach

if all_deps_found
    backend_lst += 'METAL'
    cfg_lst.set('JETSTREAM_MODULE_MULTIPLY_CONSTANT_METAL_AVAILABLE', true)
    src_lst += files([
        'base.cc',
    ])
endif
