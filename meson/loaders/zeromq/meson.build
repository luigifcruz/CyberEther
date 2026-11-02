deps = []
all_deps_found = not jst_is_browser

if all_deps_found
    deps += [
        subproject(
            'libzmq',
            required: true,
            default_options: ['default_library=static'],
        ).get_variable('libzmq_dep')
    ]

    foreach x_dep : deps
        all_deps_found = all_deps_found and x_dep.found()
    endforeach
endif

if all_deps_found
    cfg_lst.set('JETSTREAM_LOADER_ZEROMQ_AVAILABLE', true)
    dep_lst += deps
endif

ldr_lst += {'ZeroMQ': all_deps_found}
