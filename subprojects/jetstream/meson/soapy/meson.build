deps = [
    dependency('libusb', required: get_option('SOAPYSDR') or meson.is_cross_build()),
    dependency('soapysdr', 'SoapySDR', required: get_option('SOAPYSDR') or meson.is_cross_build()),
    dependency('soapyremote', 'SoapyRemote', required: get_option('SOAPYSDR') or meson.is_cross_build()),
    dependency('libairspy', required: get_option('SOAPYSDR') or meson.is_cross_build()),
    dependency('soapyairspy', required: get_option('SOAPYSDR') or meson.is_cross_build()),
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and x_dep.found()
endforeach

if all_deps_found
    cfg_lst.set('JETSTREAM_LOADER_SOAPY_AVAILABLE', true)
    dep_lst += deps
endif

ldr_lst += {'SoapySDR': all_deps_found}