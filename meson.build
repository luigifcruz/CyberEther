project(
    'cyberether', 'cpp',
    version: '0.0.0',
    default_options: ['cpp_std=c++17', 'prefix=/usr', 'optimization=3']
)

dep_lst = [
    dependency('threads'),
    subproject('samurai').get_variable('libsamurai_dep'),
    #subproject('spectrum').get_variable('libspectrum_dep'),
    subproject('render').get_variable('librender_dep'),
    subproject('jetstream').get_variable('libjetstream_dep'),
]

executable(
    'cyberether', 'main.cc',
    dependencies: dep_lst,
    install: true,
)
