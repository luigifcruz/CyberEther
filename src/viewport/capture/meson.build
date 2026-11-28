if cfg_lst.get('JETSTREAM_BACKEND_VULKAN_AVAILABLE', false)
    src_lst += files([
        'vulkan.cc',
    ])

    if jst_is_linux
        src_lst += files('vulkan_export_buffer.cc')
    endif

    if jst_is_windows
        src_lst += files('vulkan_export_d3d11.cc')
    endif
endif

if jst_is_windows
    dep_lst += [
        cpp.find_library('d3d11', required: true),
        cpp.find_library('dxgi', required: true),
    ]
endif

if cfg_lst.get('JETSTREAM_BACKEND_METAL_AVAILABLE', false)
    src_lst += files('metal.cc')
endif

instance_remote_deps = [
    'JETSTREAM_LOADER_FFMPEG_AVAILABLE',
    'JETSTREAM_LOADER_LIBDATACHANNEL_AVAILABLE',
    'JETSTREAM_LOADER_OPUS_AVAILABLE',
    'JETSTREAM_LOADER_CPPHTTPLIB_AVAILABLE',
]

instance_remote_deps_found = py.found()
foreach x_dep : instance_remote_deps
    instance_remote_deps_found = instance_remote_deps_found and cfg_lst.get(x_dep, false)
endforeach

if instance_remote_deps_found
    cfg_lst.set('JETSTREAM_INSTANCE_REMOTE_AVAILABLE', true)
    src_lst += files([
        '../../instance_remote_impl.cc',
        '../../instance_remote_encoder.cc',
        '../../instance_remote_socket.cc',
        '../../instance_remote_broker.cc',
        '../../instance_remote_audio.cc',
    ])
    if cfg_lst.get('JETSTREAM_LOADER_FFMPEG_CUDA_AVAILABLE', false)
        src_lst += files('../../instance_remote_encoder_cuda.cc')
    endif
    if jst_is_linux
        src_lst += files('../../instance_remote_encoder_vaapi.cc')
    endif
    if jst_is_windows
        src_lst += files('../../instance_remote_encoder_d3d11.cc')
    endif
    src_lst += custom_target(
        output : ['remote_viewer.cc'],
        command : [py, files(meson.project_source_root() / 'resources/remote/fetch.py'),
                   'https://cdn.cyberether.org/remote/index.html.gz', '@OUTPUT@', 'RemoteViewerPageGzip'],
    )
else
    src_lst += files([
        '../../instance_remote_stub.cc',
    ])
endif
