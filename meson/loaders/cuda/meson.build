deps = [
    dependency('cuda', version : '>=10', modules: ['cuda', 'nvidia-ml'], required: false),
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and x_dep.found()
endforeach
all_deps_found = all_deps_found

if all_deps_found
    add_languages('cuda')
    cfg_lst.set('JETSTREAM_LOADER_CUDA_AVAILABLE', true)
    dep_lst += deps
endif

ldr_lst += {'CUDA': all_deps_found}
