project(
    'metal', 'cpp',
    version: '0.0.0',
    default_options: [
        'cpp_std=c++17',
    ]
)

cfg_lst = configuration_data()
src_lst = []
dep_lst = [
  subproject('jetstream').get_variable('libjetstream_dep'),
  dependency('benchmark'),
]
inc_lst = [
    include_directories(get_option('prefix') + '/include'),
    include_directories('include'),
    include_directories('.'),
]

summary({'cpp_std': get_option('cpp_std'),
         'prefix': get_option('prefix'),
         'buildtype': get_option('buildtype'),
         'crossbuild': meson.is_cross_build(),
        }, section: 'General', bool_yn: true)

executable(
    'init',
    [src_lst, 'init.cc'],
    include_directories: inc_lst,
    dependencies: dep_lst
)
