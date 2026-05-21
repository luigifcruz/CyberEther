deps = []
all_deps_found = not jst_is_browser

if all_deps_found
    openssl_project = subproject('openssl', required: true, default_options: [
        'default_library=static',
    ])

    openssl_dep = openssl_project.get_variable('openssl_dep')
    libssl_dep = openssl_project.get_variable('libssl_dep')
    libcrypto_dep = openssl_project.get_variable('libcrypto_dep')

    deps = [
        openssl_dep,
        libssl_dep,
        libcrypto_dep,
    ]

    foreach x_dep : deps
        all_deps_found = all_deps_found and x_dep.found()
    endforeach
endif

if all_deps_found
    cfg_lst.set('JETSTREAM_LOADER_OPENSSL_AVAILABLE', true)
    dep_lst += deps
endif

ldr_lst += {'OpenSSL': all_deps_found}
