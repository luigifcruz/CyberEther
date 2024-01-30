metal_fmrk = [
    'MetalKit',
    'Metal',
    'Foundation',
]

deps = [
    dependency('appleframeworks', modules: metal_fmrk, required: false),
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

if target_machine.cpu_family() != 'aarch64'
    warning('Metal backend is only available on Apple Silicon Macs.')
    all_deps_found = false
endif

if all_deps_found
    cfg_lst.set('JETSTREAM_LOADER_METAL_AVAILABLE', true)
    dep_lst += deps
endif

ldr_lst += {'Metal': all_deps_found}
