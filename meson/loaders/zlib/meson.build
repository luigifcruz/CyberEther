deps = [
    dependency('zlib', required: false, fallback: ['zlib', 'zlib_dep']),
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and x_dep.found()
endforeach

if all_deps_found
    cfg_lst.set('JETSTREAM_LOADER_ZLIB_AVAILABLE', true)
    dep_lst += deps
endif

ldr_lst += {'zlib': all_deps_found}
