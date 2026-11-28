remote_opt = get_option('remote')

deps = []
all_deps_found = remote_opt.enabled() and not jst_is_browser

if all_deps_found
    deps += subproject('opus', required: true, default_options: [
        'buildtype=' + get_option('buildtype'),
        'default_library=static',
        'extra-programs=disabled',
        'tests=disabled',
        'docs=disabled',
    ]).get_variable('opus_dep')

    deps += declare_dependency(
        link_args: meson.get_compiler('c').get_supported_link_arguments('-Wl,--exclude-libs,libopus.a'),
    )

    foreach x_dep : deps
        all_deps_found = all_deps_found and x_dep.found()
    endforeach
endif

if all_deps_found
    cfg_lst.set('JETSTREAM_LOADER_OPUS_AVAILABLE', true)
    dep_lst += deps
endif

ldr_lst += {'Opus': all_deps_found}
