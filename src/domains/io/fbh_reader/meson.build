if cfg_lst.get('JETSTREAM_LOADER_HDF5_AVAILABLE', false)
    cfg_lst.set('JETSTREAM_MODULE_FBH_READER_AVAILABLE', true)

    src_lst += files([
        'block_impl.cc',
        'module_impl.cc',
        'module_impl_native_cpu.cc',
    ])

    sum_lst += {'FbhReader': ['CPU']}
endif
