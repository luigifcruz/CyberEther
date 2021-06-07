project(
    'jetstream', 'cpp',
    version: '0.0.0',
    yielding: false,
    default_options: [
        'cpp_std=c++17',
        'cuda_std=c++17',
    ]
)

summary({'cpp_std': get_option('cpp_std'),
         'prefix': get_option('prefix'),
         'buildtype': get_option('buildtype'),
        }, section: 'General')

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

subdir('src')
subdir('include')

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