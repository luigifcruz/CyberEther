project(
    'cyberether', 'cpp',
    version: '0.0.0',
    default_options: [
        'cpp_std=c++17',
        'cuda_std=c++14',
    ]
)

summary({'cpp_std': get_option('cpp_std'),
         'prefix': get_option('prefix'),
         'buildtype': get_option('buildtype'),
        }, section: 'General')

dep_lst = [
    dependency('threads'),
    subproject('samurai').get_variable('libsamurai_dep'),
    subproject('render').get_variable('librender_dep'),
    subproject('jetstream').get_variable('libjetstream_dep'),
]

executable(
    'cyberether', 'main.cc',
    dependencies: dep_lst,
    install: true,
)