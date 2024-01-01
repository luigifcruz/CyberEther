deps = [
    dependency('soapysdr', 'SoapySDR', required: get_option('SOAPYSDR')),
]

if jst_is_ios
    deps += [
        dependency('soapysdr', 'SoapySDR', required: true),
        dependency('soapyremote', 'SoapyRemote', required: true),
    ]
endif

if jst_is_browser or jst_is_windows
    deps += [
        dependency('soapysdr', 'SoapySDR', required: true),
        dependency('libairspy', required: true),
        dependency('soapyairspy', required: true),
        dependency('librtlsdr', required: true),
        dependency('soapyrtlsdr', required: true),  
    ]
endif

if jst_is_browser
    deps += [
        dependency('soapyremote', 'SoapyRemote', required: true),
        dependency('libusb-browser', required: true),
    ]
endif

if jst_is_windows
    deps += [
        dependency('libusb', fallback: 'libusb-1.0', required: true),
    ]
endif

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and x_dep.found()
endforeach

if all_deps_found
    cfg_lst.set('JETSTREAM_LOADER_SOAPY_AVAILABLE', true)
    dep_lst += deps
endif

ldr_lst += {'SoapySDR': all_deps_found}