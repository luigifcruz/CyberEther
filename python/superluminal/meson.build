deps = [
    subproject('nanobind').get_variable('nanobind_dep'),
    py.dependency()
]

if get_option('buildtype') == 'debugoptimized' and not jst_is_windows
    run_command(
        find_program('bash'),
        '-c',
        '@0@ -fs @1@/* @2@'.format(
            find_program('ln').full_path(),
            meson.current_source_dir(),
            meson.current_build_dir()
        ),
        check: true
    )
endif

py.extension_module(
    '_impl',
    sources: 'superluminal.cc',
    include_directories: inc_lst,
    dependencies: deps + dep_lst,
    link_with: libjetstream,
    install: true,
    install_dir: py_install_path,
)
