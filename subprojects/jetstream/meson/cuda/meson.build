deps = [
    dependency('cuda', version : '>=10', modules: ['dl'], required: get_option('CUDA'))
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and x_dep.found()
endforeach
all_deps_found = all_deps_found and not meson.is_cross_build()

if all_deps_found
    add_languages('cuda')
    cfg_lst.set('JETSTREAM_CUDA_AVAILABLE', true)
    dep_lst += deps
endif

ldr_lst += {'CUDA': all_deps_found}
