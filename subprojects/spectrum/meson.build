project(
    'spectrum', 'cpp',
    version: '0.0.0',
    default_options: ['cpp_std=c++17', 'prefix=/usr', 'optimization=3']
)

cfg_lst = configuration_data()
src_lst = []
dep_lst = []
inc_lst = [
    include_directories(get_option('prefix') + '/include'),
    include_directories('include'),
    include_directories('.'),
]

subdir('src')
subdir('include')

libspectrum = library(
    'spectrum',
    src_lst,
    include_directories: inc_lst,
    dependencies: dep_lst,
    install: true
)

libspectrum_dep = declare_dependency(
    include_directories: inc_lst,
    link_with: libspectrum
)