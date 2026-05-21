plugin_sources = []

subdir('blueprint/src')

shared_module(
    'cyberether_blueprint_plugin',
    plugin_sources,
    include_directories: include_directories('blueprint/include', 'blueprint/src'),
    dependencies: libjetstream_dep,
    gnu_symbol_visibility: 'hidden',
    name_prefix: '',
    build_by_default: true,
    install: false,
)

sum_lst += {'Plugin blueprint': true}
