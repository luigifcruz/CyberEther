deps = [
    dependency('whisper', required: false),
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and x_dep.found()
endforeach

if all_deps_found
    cfg_lst.set('JETSTREAM_LOADER_WHISPERCPP_AVAILABLE', true)
    dep_lst += deps
endif

ldr_lst += {'whisper.cpp': all_deps_found}
