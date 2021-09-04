project(
    'render', 'cpp',
    version: '0.0.0',
    default_options: [
        'cpp_std=c++17',
    ]
)

cfg_lst = configuration_data()
src_lst = []
dep_lst = [
    subproject('jetstream').get_variable('libjetstream_dep'),
]
inc_lst = [
    include_directories(get_option('prefix') + '/include'),
    include_directories('include/render/tools'),
    include_directories('include'),
    include_directories('.'),
]

subdir('meson')
subdir('src')
subdir('include')

summary({'cpp_std': get_option('cpp_std'),
         'prefix': get_option('prefix'),
         'buildtype': get_option('buildtype'),
         'CUDA interop': cuda_dep.found() and not meson.is_cross_build(),
         'crossbuild': meson.is_cross_build(),
        }, section: 'General', bool_yn: true)

librender = library(
    'render',
    src_lst,
    include_directories: inc_lst,
    dependencies: dep_lst,
    install: true
)

librender_dep = declare_dependency(
    include_directories: inc_lst,
    dependencies: dep_lst,
    link_with: librender
)

pkg = import('pkgconfig')
pkg.generate(
  librender,
  libraries: dep_lst,
)
