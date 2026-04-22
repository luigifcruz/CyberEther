project('libmodes', 'c')

libmodes_src = files('src/mode-s.c')

libmodes_c_args = []
if meson.get_compiler('c').get_id() != 'msvc'
    libmodes_c_args += ['-Wno-unused-variable', '-Wno-sign-compare']
endif

libmodes_lib = static_library(
    'modes',
    libmodes_src,
    include_directories: include_directories('include'),
    c_args: libmodes_c_args,
)

libmodes_dep = declare_dependency(
    include_directories: include_directories('include'),
    link_with: libmodes_lib,
)
