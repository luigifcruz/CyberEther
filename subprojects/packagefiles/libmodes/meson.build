project('libmodes', 'c')

libmodes_src = files('src/mode-s.c')

libmodes_lib = static_library(
    'modes',
    libmodes_src,
    include_directories: include_directories('include'),
    c_args: ['-Wno-unused-variable', '-Wno-sign-compare'],
)

libmodes_dep = declare_dependency(
    include_directories: include_directories('include'),
    link_with: libmodes_lib,
)
