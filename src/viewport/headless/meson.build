if cfg_lst.get('JETSTREAM_BACKEND_VULKAN_AVAILABLE', false)
    src_lst += files([
        'vulkan.cc',
    ])
    sum_lst += { 'Headless (Vulkan)' : true }
endif

if cfg_lst.get('JETSTREAM_BACKEND_METAL_AVAILABLE', false)
    src_lst += files('metal.cc')
    sum_lst += { 'Headless (Metal)' : true }
endif
