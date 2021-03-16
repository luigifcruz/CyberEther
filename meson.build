project(
    'cyberether', 'cpp',
    version: '0.0.0',
    default_options: ['cpp_std=c++17', 'prefix=/usr', 'optimization=3']
)

dep_lst = [
    subproject('spectrum').get_variable('libspectrum_dep'),
    subproject('render').get_variable('librender_dep'),
]

executable(
    'cyberether', 'main.cc',
    dependencies: dep_lst,
    install: true,
)