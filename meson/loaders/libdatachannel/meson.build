remote_opt = get_option('remote')

deps = []
all_deps_found = remote_opt.enabled() and \
                 not jst_is_browser and \
                 cfg_lst.get('JETSTREAM_LOADER_OPENSSL_AVAILABLE', false)

if all_deps_found
    subproject('libsrtp2', required: true, default_options: [
        'default_library=static',
        'crypto-library=openssl',
        'doc=disabled',
        'pcap-tests=disabled',
        'tests=disabled',
    ])

    subproject('usrsctp', required: true, default_options: [
        'default_library=static',
        'sctp_build_programs=false',
        'sctp_inet=false',
        'sctp_inet6=false',
    ])

    deps += [
        subproject('libdatachannel', required: true).get_variable('libdatachannel_dep'),
    ]

    foreach x_dep : deps
        all_deps_found = all_deps_found and x_dep.found()
    endforeach
endif

if all_deps_found
    cfg_lst.set('JETSTREAM_LOADER_LIBDATACHANNEL_AVAILABLE', true)
    dep_lst += deps
endif

ldr_lst += {'libdatachannel': all_deps_found}
