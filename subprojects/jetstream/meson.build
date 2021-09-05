project(
    'jetstream', 'cpp',
    version: '0.0.0',
    default_options: [
        'cpp_std=c++17',
    ]
)

cfg_lst = configuration_data()
src_lst = []
dep_lst = [
  dependency('threads'),
]
inc_lst = [
    include_directories(get_option('prefix') + '/include'),
    include_directories('include'),
    include_directories('.'),
]

is_release = false
if get_option('buildtype') == 'release'
  is_release = true
endif
cfg_lst.set('JETSTREAM_DEBUG', not is_release)

cuda_dep = dependency('cuda', version : '>=11', modules: ['dl'],  required: false)
if cuda_dep.found() and not meson.is_cross_build()
  cfg_lst.set('JETSTREAM_CUDA_AVAILABLE', true)
  dep_lst += cuda_dep
endif

subdir('include')
subdir('src')

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

pkg = import('pkgconfig')
pkg.generate(
  libjetstream,
  libraries: dep_lst,
)
