metal_fmrk = [
    'MetalKit',
    'Metal',
    'Foundation',
]

deps = [
    dependency('appleframeworks', modules: metal_fmrk, required: false),
]

progs = [
    find_program('glslangValidator', required: false),
    find_program('spirv-cross', required: false),
    find_program('python3', 'python', required: false),
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and x_dep.found()
endforeach
foreach x_prog : progs
    all_deps_found = all_deps_found and x_prog.found()
endforeach

if all_deps_found
    cfg_lst.set('JETSTREAM_LOADER_METAL_AVAILABLE', true)
    dep_lst += deps
endif

ldr_lst += {'Metal': all_deps_found}
