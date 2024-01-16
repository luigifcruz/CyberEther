sum_lst = {}

# Hello World

deps = [
    'JETSTREAM_MODULE_FFT_CPU_AVAILABLE',
    'JETSTREAM_MODULE_SOAPY_CPU_AVAILABLE',
    'JETSTREAM_MODULE_WINDOW_CPU_AVAILABLE',
    'JETSTREAM_MODULE_MULTIPLY_CPU_AVAILABLE',
    'JETSTREAM_MODULE_AMPLITUDE_CPU_AVAILABLE',
    'JETSTREAM_MODULE_SCALE_CPU_AVAILABLE',
    'JETSTREAM_MODULE_LINEPLOT_CPU_AVAILABLE',
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and cfg_lst.get(x_dep, false)
endforeach

if all_deps_found
    executable(
        'hello_world', 
        'hello_world.cc',
        dependencies: libjetstream_dep,
        install: false,
    )
endif

sum_lst += {'Hello World': all_deps_found}

# Custom Interfaces

deps = [
    'JETSTREAM_BACKEND_VULKAN_AVAILABLE',
    'JETSTREAM_VIEWPORT_GLFW_AVAILABLE',
    'JETSTREAM_MODULE_FFT_CPU_AVAILABLE',
    'JETSTREAM_MODULE_SOAPY_CPU_AVAILABLE',
    'JETSTREAM_MODULE_WINDOW_CPU_AVAILABLE',
    'JETSTREAM_MODULE_MULTIPLY_CPU_AVAILABLE',
    'JETSTREAM_MODULE_AMPLITUDE_CPU_AVAILABLE',
    'JETSTREAM_MODULE_SCALE_CPU_AVAILABLE',
    'JETSTREAM_MODULE_LINEPLOT_CPU_AVAILABLE',
    'JETSTREAM_MODULE_WATERFALL_CPU_AVAILABLE',
    'JETSTREAM_MODULE_SPECTROGRAM_CPU_AVAILABLE',
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and cfg_lst.get(x_dep, false)
endforeach

if all_deps_found
    executable(
        'custom_interface', 
        'custom_interface.cc',
        dependencies: libjetstream_dep,
        install: false,
    )
endif

sum_lst += {'Custom Interfaces': all_deps_found}

summary(sum_lst, section: 'Examples', bool_yn: true)