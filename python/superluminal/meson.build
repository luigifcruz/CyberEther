cmake = import('cmake')
opt_var = cmake.subproject_options()
if get_option('buildtype') == 'debugoptimized'
    opt_var.add_cmake_defines({'CMAKE_BUILD_TYPE': 'Debug'})
endif
opt_var.append_compile_args('cpp', '-UNB_ABORT_ON_LEAK')
sub_proj = cmake.subproject('nanobind', options: opt_var)

deps = []
deps += sub_proj.dependency('nanobind-static')
deps += py.dependency()

if get_option('buildtype') == 'debugoptimized'
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
