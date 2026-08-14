deps = [
    'JETSTREAM_SUPERLUMINAL_AVAILABLE',
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and cfg_lst.get(x_dep, false)
endforeach

if all_deps_found
    executable(
        'global_interface',
        'global_interface.cc',
        dependencies: libjetstream_dep,
        install: false,
    )
endif

sum_lst += {'Superluminal | Global Interface': all_deps_found}
