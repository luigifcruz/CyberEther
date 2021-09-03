project(
    'gadget', 'cpp',
    version: '0.0.0',
    default_options: [
        'cpp_std=c++17',
        'cuda_std=c++17',
    ]
)

cfg_lst = configuration_data()
src_lst = []
dep_lst = [
    dependency('cuda', version : '>=10', required: false),
]
inc_lst = [
    include_directories(get_option('prefix') + '/include'),
    include_directories('include'),
    include_directories('.'),
]

subdir('src')
subdir('include')

summary({'cpp_std': get_option('cpp_std'),
         'prefix': get_option('prefix'),
         'buildtype': get_option('buildtype'),
         'crossbuild': meson.is_cross_build(),
        }, section: 'General', bool_yn: true)

libgadget = library(
    'gadget',
    src_lst,
    include_directories: inc_lst,
    dependencies: dep_lst,
    install: true
)

libgadget_dep = declare_dependency(
    include_directories: inc_lst,
    dependencies: dep_lst,
    link_with: libgadget
)

pkg = import('pkgconfig')
pkg.generate(
  libgadget,
  libraries: dep_lst,
)
