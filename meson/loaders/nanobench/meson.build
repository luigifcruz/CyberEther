deps = [
    subproject('nanobench', required: true).get_variable('nanobench_dep'),
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and x_dep.found()
endforeach

if all_deps_found
    cfg_lst.set('JETSTREAM_LOADER_NANOBENCH_AVAILABLE', true)
    dep_lst += deps
endif

ldr_lst += {'Nanobench': all_deps_found}
