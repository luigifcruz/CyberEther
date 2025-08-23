deps = [
    subproject(
        'json',
        default_options: []
    ).get_variable('nlohmann_json_dep'),
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and x_dep.found()
endforeach
all_deps_found = all_deps_found or jst_is_browser

if all_deps_found
    cfg_lst.set('JETSTREAM_LOADER_JSON_AVAILABLE', true)
    dep_lst += deps
endif

ldr_lst += {'JSON': all_deps_found}
