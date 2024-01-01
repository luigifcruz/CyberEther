deps = [
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
all_deps_found = all_deps_found and jst_is_browser

if all_deps_found
    cfg_lst.set('JETSTREAM_LOADER_WEBGPU_AVAILABLE', true)
    dep_lst += deps
endif

ldr_lst += {'WebGPU': all_deps_found}