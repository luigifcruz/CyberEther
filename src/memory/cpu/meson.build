backend_lst = []

deps = [
    'JETSTREAM_BACKEND_CPU_AVAILABLE',
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and cfg_lst.get(x_dep, false)
endforeach

if all_deps_found
    src_lst += files([
        'buffer.cc',
    ])
endif