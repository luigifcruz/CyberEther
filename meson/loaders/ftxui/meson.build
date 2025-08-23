ftxui = subproject(
    'ftxui',
    default_options: []
)

deps = [
    ftxui.get_variable('screen_dep'),
    ftxui.get_variable('dom_dep'),
    ftxui.get_variable('component_dep'),
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and x_dep.found()
endforeach
all_deps_found = all_deps_found or jst_is_browser

if all_deps_found
    cfg_lst.set('JETSTREAM_LOADER_FTXUI_AVAILABLE', true)
    dep_lst += deps
endif

ldr_lst += {'ftxui': all_deps_found}
