velopack_available = (target_machine.system() == 'linux' or jst_is_macos or jst_is_windows) and not jst_is_wheel

if velopack_available
    velopack_project = subproject('velopack', required: true)
    velopack_include = velopack_project.get_variable('velopack_include')
    velopack_link_path = velopack_project.get_variable('velopack_link_path')

    velopack_system_deps = []
    if jst_is_linux
        velopack_system_deps += [
            cpp.find_library('dl', required: true),
            cpp.find_library('m', required: true),
        ]
    elif jst_is_windows
        velopack_system_deps += [
            cpp.find_library('bcrypt', required: true),
            cpp.find_library('ntdll', required: true),
        ]
    endif

    velopack_dep = declare_dependency(
        include_directories: velopack_include,
        dependencies: velopack_system_deps,
        link_args: [velopack_link_path],
    )
    dep_lst += [velopack_dep]
    cfg_lst.set('JETSTREAM_LOADER_VELOPACK_AVAILABLE', true)
endif

ldr_lst += {'Velopack': velopack_available}
