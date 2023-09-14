deps = [
    dependency('soapysdr', 'SoapySDR', required: get_option('SOAPYSDR') or meson.is_cross_build()),
]

if target_machine.system() == 'ios'
    deps += [
        dependency('soapyremote', 'SoapyRemote', required: meson.is_cross_build()),
    ]
endif

if target_machine.system() == 'emscripten'
    deps += [
        dependency('soapyremote', 'SoapyRemote', required: meson.is_cross_build()),
        dependency('libusb', required: meson.is_cross_build()),
        dependency('libairspy', required: meson.is_cross_build()),
        dependency('soapyairspy', required: meson.is_cross_build()),
        dependency('librtlsdr', required: meson.is_cross_build()),
        dependency('soapyrtlsdr', required: meson.is_cross_build()),  
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