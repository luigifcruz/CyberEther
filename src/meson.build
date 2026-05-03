subdir('backend')
subdir('memory')
subdir('tools')
subdir('viewport')
subdir('render')
subdir('domains')
subdir('platform')
subdir('compositor')
subdir('superluminal')

if jst_is_browser
    src_lst += files('app_browser.cc')
endif

if jst_is_linux or \
   jst_is_macos or \
   jst_is_windows
    src_lst += files('app_native.cc')
endif

src_lst += files([
    'flowgraph.cc',
    'settings.cc',
    'parser_map.cc',
    'parser_decode.cc',
    'parser_encode.cc',
    'parser_yaml.cc',
    'tensor_link.cc',
    'instance.cc',
    'instance_remote.cc',
    'instance_remote_supervisor.cc',
    'logger.cc',
    'module.cc',
    'module_impl.cc',
    'module_context.cc',
    'module_interface.cc',
    'module_surface.cc',
    'block.cc',
    'block_impl.cc',
    'block_interface.cc',
    'registry.cc',
    'benchmark.cc',
    'testing.cc',
    'compositor.cc',
    'compositor_impl.cc',
    'runtime.cc',
    'runtime_impl_native_cpu.cc',
    'runtime_context_native_cpu.cc',
    'scheduler.cc',
    'scheduler_context.cc',
    'scheduler_synchronous.cc',
])

if cfg_lst.get('JETSTREAM_BACKEND_CUDA_AVAILABLE', false)
    src_lst += files([
        'runtime_impl_native_cuda.cc',
        'runtime_context_native_cuda.cc',
    ])
endif
