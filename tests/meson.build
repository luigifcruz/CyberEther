deps = [
    'JETSTREAM_BACKEND_METAL_AVAILABLE',
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and cfg_lst.get(x_dep, false)
endforeach

if all_deps_found
    test('memory', executable(
        'jetstream-memory', 'memory.cc',
        dependencies: libjetstream_dep,
    ), is_parallel: false, timeout: 0)
endif