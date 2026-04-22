deps = []
all_deps_found = not (jst_is_browser or jst_is_windows)

if all_deps_found
    deps += [
        subproject(
            'cpp-httplib',
            required: true,
            default_options: [
                'tls=auto',
                'compile=true',
                'default_library=static'
            ]
        ).get_variable('cpp_httplib_dep')
    ]

    foreach x_dep : deps
        all_deps_found = all_deps_found and x_dep.found()
    endforeach
endif

if all_deps_found
    cfg_lst.set('JETSTREAM_LOADER_CPPHTTPLIB_AVAILABLE', true)
    dep_lst += deps
endif

ldr_lst += {'cpp-httplib': all_deps_found}
