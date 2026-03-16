deps = [
    'JETSTREAM_SUPERLUMINAL_AVAILABLE',
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and cfg_lst.get(x_dep, false)
endforeach

if all_deps_found
    executable(
        'remote',
        'remote.cc',
        dependencies: libjetstream_dep,
        install: false,
    )

    if cfg_lst.get('JETSTREAM_SUPERLUMINAL_PYTHON_AVAILABLE', false)
        configure_file(
            input: 'remote.py',
            output: 'remote.py',
            copy: true,
        )
    endif
endif

sum_lst += {'Superluminal | Remote': all_deps_found}
