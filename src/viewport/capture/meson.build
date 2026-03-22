if cfg_lst.get('JETSTREAM_BACKEND_VULKAN_AVAILABLE', false)
    src_lst += files([
        'vulkan.cc',
    ])
endif

instance_remote_deps = [
    'JETSTREAM_LOADER_GSTREAMER_AVAILABLE',
    'JETSTREAM_LOADER_CPPHTTPLIB_AVAILABLE',
    'JETSTREAM_LOADER_JSON_AVAILABLE',
]

instance_remote_deps_found = true
foreach x_dep : instance_remote_deps
    instance_remote_deps_found = instance_remote_deps_found and cfg_lst.get(x_dep, false)
endforeach

if instance_remote_deps_found
    cfg_lst.set('JETSTREAM_INSTANCE_REMOTE_AVAILABLE', true)
    src_lst += files([
        '../../instance_remote_impl.cc',
    ])
else
    src_lst += files([
        '../../instance_remote_stub.cc',
    ])
endif
