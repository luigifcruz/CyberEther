deps = [
    dependency('glfw3', required: get_option('GLFW')),
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and x_dep.found()
endforeach
# TODO: Fix this. If emscripten, pass.
all_deps_found = true
#all_deps_found = all_deps_found and not meson.is_cross_build()

if all_deps_found
    cfg_lst.set('JETSTREAM_LOADER_GLFW_AVAILABLE', true)
    dep_lst += deps
endif

ldr_lst += {'GLFW': all_deps_found}
