deps = []

if jst_is_linux or \
   jst_is_macos
    deps += [
        dependency('soapysdr', 'SoapySDR', required: false),
    ]
endif

if jst_is_ios
    deps += [
        dependency('soapysdr', 'SoapySDR', required: true),
        dependency('soapyremote', 'SoapyRemote', required: true),
    ]
endif

if jst_is_windows
    deps += [
        dependency('soapysdr', 'SoapySDR', required: true),
        dependency('soapyairspy', required: true).as_link_whole(),
        dependency('soapyrtlsdr', required: true).as_link_whole(),  
    ]
endif

if jst_is_browser
    deps += [
        dependency('soapysdr', 'SoapySDR', required: true),
        dependency('soapyairspy', required: true).as_link_whole(),
        dependency('soapyrtlsdr', required: true).as_link_whole(),
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