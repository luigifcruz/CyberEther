deps = []
all_deps_found = true

soapy_static_options = [
    'default_library=static',
]

if not jst_is_android
    soapysdr_dep = subproject(
        'soapysdr',
        required: true,
        default_options: soapy_static_options,
    ).get_variable('soapysdr_dep')
endif

if jst_is_ios
    deps += [
        soapysdr_dep,
        subproject(
            'soapyremote',
            required: true,
            default_options: soapy_static_options,
        ).get_variable('soapyremote_dep'),
    ]
endif

if jst_is_macos or jst_is_linux
    deps += [
        soapysdr_dep,
    ]
    whole_dep_lst += [
        subproject(
            'soapyairspy',
            required: true,
            default_options: soapy_static_options,
        ).get_variable('soapyairspy_dep').as_link_whole(),
        subproject(
            'limesuite',
            required: true,
            default_options: soapy_static_options,
        ).get_variable('soapylms7_dep').as_link_whole(),
        subproject(
            'soapyrtlsdr',
            required: true,
            default_options: soapy_static_options,
        ).get_variable('soapyrtlsdr_dep').as_link_whole(),
    ]
endif

if jst_is_windows
    deps += [
        soapysdr_dep,
    ]
    whole_dep_lst += [
        subproject(
            'soapyrtlsdr',
            required: true,
            default_options: soapy_static_options,
        ).get_variable('soapyrtlsdr_dep').as_link_whole(),
        subproject(
            'limesuite',
            required: true,
            default_options: soapy_static_options,
        ).get_variable('soapylms7_dep').as_link_whole(),
    ]
endif

if jst_is_browser
    deps += [
        soapysdr_dep,
    ]
    whole_dep_lst += [
        subproject(
            'soapyairspy',
            required: true,
            default_options: soapy_static_options,
        ).get_variable('soapyairspy_dep').as_link_whole(),
        subproject(
            'limesuite',
            required: true,
            default_options: soapy_static_options,
        ).get_variable('soapylms7_dep').as_link_whole(),
        subproject(
            'soapyrtlsdr',
            required: true,
            default_options: soapy_static_options,
        ).get_variable('soapyrtlsdr_dep').as_link_whole(),
    ]
endif

if jst_is_android
    # TODO: Implement SoapySDR for Android.
    all_deps_found = false
endif

foreach x_dep : deps
    all_deps_found = all_deps_found and x_dep.found()
endforeach

if all_deps_found
    cfg_lst.set('JETSTREAM_LOADER_SOAPY_AVAILABLE', true)
    dep_lst += deps
endif

ldr_lst += {'SoapySDR': all_deps_found}
