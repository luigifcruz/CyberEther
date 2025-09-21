deps = [
    'JETSTREAM_SUPERLUMINAL_AVAILABLE',
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and cfg_lst.get(x_dep, false)
endforeach

if all_deps_found
    executable(
        'interface',
        'interface.cc',
        dependencies: libjetstream_dep,
        install: false,
    )

    if cfg_lst.get('JETSTREAM_SUPERLUMINAL_PYTHON_AVAILABLE', false)
        configure_file(
            input: 'interface.py',
            output: 'interface.py',
            copy: true,
        )
    endif
endif

sum_lst += {'Superluminal | Interface': all_deps_found}
