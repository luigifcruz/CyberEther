deps = [
    subproject('render').get_variable('librender_dep'),
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and x_dep.found()
endforeach
all_deps_found = all_deps_found and not meson.is_cross_build()

if all_deps_found
    cfg_lst.set('JETSTREAM_RENDER_AVAILABLE', true)
    dep_lst += deps
endif

ldr_lst += {'Render': all_deps_found}
