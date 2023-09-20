executable(
    'hello_world', 
    'hello_world.cc',
    dependencies: libjetstream_dep,
    install: false,
)

if cfg_lst.get('JETSTREAM_BACKEND_VULKAN_AVAILABLE', false)
    executable(
        'custom_interface', 
        'custom_interface.cc',
        dependencies: libjetstream_dep,
        install: false,
    )
endif