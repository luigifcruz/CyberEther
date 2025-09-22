src_lst += files([
    'buffer_cpu.cc',
    'buffer_metal.cc',
    'buffer.cc',
    'tensor.cc',
    'token.cc',
    'types.cc',
])

if cfg_lst.get('JETSTREAM_BACKEND_VULKAN_AVAILABLE', false)
    src_lst += files([
        'buffer_vulkan.cc',
    ])
endif
