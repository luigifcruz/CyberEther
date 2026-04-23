deps = [
    subproject('libmodes', required: true).get_variable('libmodes_dep')
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and x_dep.found()
endforeach

if all_deps_found
    cfg_lst.set('JETSTREAM_LOADER_LIBMODES_AVAILABLE', true)
    dep_lst += deps
endif

ldr_lst += {'Libmodes': all_deps_found}
