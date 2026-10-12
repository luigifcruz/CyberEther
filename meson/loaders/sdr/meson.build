sdr_static_options = ['default_library=static']
sdr_found = false

if jst_is_macos or jst_is_linux or jst_is_windows or jst_is_browser
    sdr_families = {
        'RTLSDR': ['librtlsdr', 'librtlsdr_dep'],
        'AIRSPY': ['libairspy', 'libairspy_dep'],
        'AIRSPYHF': ['libairspyhf', 'libairspyhf_dep'],
        'HACKRF': ['libhackrf', 'libhackrf_dep'],
        'LIMESUITE': ['limesuite', 'limesuite_dep'],
        'BLADERF': ['libbladerf', 'libbladerf_dep'],
    }

    foreach family, source : sdr_families
        family_dep = subproject(
            source[0],
            required: true,
            default_options: sdr_static_options,
        ).get_variable(source[1])

        if family_dep.found()
            cfg_lst.set('JETSTREAM_LOADER_' + family + '_AVAILABLE', true)
            dep_lst += family_dep
            sdr_found = true
        endif
        ldr_lst += {family: family_dep.found()}
    endforeach

    if jst_is_linux and not jst_is_static
        lib_link_args += [
            '-Wl,--exclude-libs,liblibrtlsdr.a:liblibairspy.a:liblibairspyhf.a:libhackrf.a:' +
            'libLimeSuite.a:' +
            'libbladeRF.a:libusb-1.0.a',
        ]
    endif

    if jst_is_macos and not jst_is_static
        foreach symbols : ['_rtlsdr_*', '_airspy_*', '_airspyhf_*', '_hackrf_*', '_bladerf_*', '_LMS_*',
                           '_libusb_*', '__ZN4lime*']
            lib_link_args += '-Wl,-unexported_symbol,' + symbols
        endforeach
    endif
endif

if sdr_found
    cfg_lst.set('JETSTREAM_LOADER_SDR_AVAILABLE', true)
endif

ldr_lst += {'SDR': sdr_found}
