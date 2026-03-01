# Superluminal requires the CPU backend and core modules (FFT, amplitude,
# scale, lineplot, etc.) which are always compiled when CPU is available.
if cfg_lst.get('JETSTREAM_BACKEND_CPU_AVAILABLE', false)
    cfg_lst.set('JETSTREAM_SUPERLUMINAL_AVAILABLE', true)

    src_lst += files([
        'base.cc',
        'dmi_module_impl.cc',
        'dmi_module_impl_native_cpu.cc',
        'dmi_block_impl.cc',
    ])
endif
