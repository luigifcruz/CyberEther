deps = []

if not jst_is_browser
    glfw_default_options = [
        'default_library=static',
        'install=false',
    ]

    if jst_is_linux
        glfw_default_options += ['display-api=x11,wayland']
    endif

    deps += [
        subproject(
            'glfw',
            required: true,
            default_options: glfw_default_options
        ).get_variable('glfw_dep'),
    ]
endif

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and x_dep.found()
endforeach
all_deps_found = all_deps_found or jst_is_browser

if all_deps_found
    cfg_lst.set('JETSTREAM_LOADER_GLFW_AVAILABLE', true)
    dep_lst += deps
endif

ldr_lst += {'GLFW': all_deps_found}
