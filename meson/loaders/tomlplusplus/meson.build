deps = [
    subproject('tomlplusplus', required: true, default_options: [
        'default_library=static',
    ]).get_variable('tomlplusplus_dep'),
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and x_dep.found()
endforeach

if all_deps_found
    cfg_lst.set('JETSTREAM_LOADER_TOMLPLUSPLUS_AVAILABLE', true)
    dep_lst += deps
endif

ldr_lst += {'toml++': all_deps_found}
