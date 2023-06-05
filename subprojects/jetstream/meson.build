project(
    'jetstream',
    ['cpp'],
    version: '0.1.0',
    default_options: [
        'cpp_std=c++20',
        'buildtype=release',
        'b_ndebug=if-release',
    ]
)

cfg_lst = configuration_data()
src_lst = []
dep_lst = [
    dependency('threads'),
    dependency('fmt'),
]
inc_lst = [
    include_directories('include'),
    include_directories('.'),
    include_directories('include/jetstream/backend/devices/metal'),
    include_directories('include/jetstream/render/tools'),
]

subdir('meson')
subdir('shaders')
subdir('src')
subdir('include')

summary({'cpp_std': get_option('cpp_std'),
         'prefix': get_option('prefix'),
         'buildtype': get_option('buildtype'),
         'crossbuild': meson.is_cross_build(),
        }, section: 'General', bool_yn: true)

libjetstream = library(
    'jetstream',
    src_lst,
    include_directories: inc_lst,
    dependencies: dep_lst,
    install: true
)

libjetstream_dep = declare_dependency(
    include_directories: inc_lst,
    dependencies: dep_lst,
    link_with: libjetstream
)

# TODO: Re-enable tests.
#subdir('tests')

pkg = import('pkgconfig')
pkg.generate(
  libjetstream,
  libraries: dep_lst,
)
