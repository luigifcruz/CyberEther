deps = [
    dependency('vulkan', required: get_option('VULKAN')),
]

progs = [
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and x_dep.found()
endforeach
foreach x_prog : progs
    all_deps_found = all_deps_found and x_prog.found()
endforeach
all_deps_found = all_deps_found and not meson.is_cross_build()

if all_deps_found
    cfg_lst.set('JETSTREAM_LOADER_VULKAN_AVAILABLE', true)
    dep_lst += deps
endif

ldr_lst += {'Vulkan': all_deps_found}
