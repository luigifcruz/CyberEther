deps = [
    subproject('stb', required: true).get_variable('stb_dep')
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and x_dep.found()
endforeach

if all_deps_found
    cfg_lst.set('JETSTREAM_LOADER_STB_AVAILABLE', true)
    dep_lst += deps
endif

ldr_lst += {'stb': all_deps_found}
