fbh_reader_opt = get_option('fbh_reader')

if not fbh_reader_opt.disabled()
    hdf5_dep = dependency('hdf5', required: false)
    if not hdf5_dep.found()
        hdf5_dep = dependency('hdf5_serial', required: false)
    endif

    if fbh_reader_opt.enabled() and not hdf5_dep.found()
        error('fbh_reader=enabled but HDF5 was not found. ' +
              'Install with: brew install hdf5  (macOS) or ' +
              'apt install libhdf5-dev  (Linux).')
    endif

    if hdf5_dep.found()
        cfg_lst.set('JETSTREAM_LOADER_HDF5_AVAILABLE', true)
        dep_lst += [hdf5_dep]
    endif

    ldr_lst += {'HDF5': hdf5_dep.found()}
else
    ldr_lst += {'HDF5': false}
endif
