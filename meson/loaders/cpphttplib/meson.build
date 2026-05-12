deps = []
all_deps_found = not jst_is_browser and cfg_lst.get('JETSTREAM_LOADER_OPENSSL_AVAILABLE', false)

if all_deps_found
    deps += [
        subproject(
            'cpp-httplib',
            required: true,
            default_options: [
                'tls=enabled',
                'tls_backend=openssl',
                'brotli=disabled',
                'compile=true',
                'default_library=static',
                'zstd=disabled',
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
