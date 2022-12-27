project(
    'cyberether', 'cpp',
    version: '0.0.0',
    default_options: [
        'cpp_std=c++20',
        'b_asneeded=false',
    ]
)

summary({'cpp_std': get_option('cpp_std'),
         'prefix': get_option('prefix'),
         'buildtype': get_option('buildtype'),
         'crossbuild': meson.is_cross_build(),
        }, section: 'General', bool_yn: true)

dep_lst = [
    dependency('SoapySDR'),
    dependency('threads'),
    subproject('jetstream').get_variable('libjetstream_dep'),
]

executable(
    'cyberether', 'main.cc',
    dependencies: dep_lst,
    install: true,
)
