if cfg_lst.get('JETSTREAM_BACKEND_VULKAN_AVAILABLE', false)
    src_lst += files([
        'vulkan.cc',
    ])
endif

instance_remote_deps = [
    'JETSTREAM_LOADER_GSTREAMER_AVAILABLE',
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
        '../../instance_remote_socket.cc',
        '../../instance_remote_broker.cc',
    ])
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
