plugin_sources = []
plugin_test_lst = []
plugin_example = files('blueprint/examples/blueprint_gain.yml')
plugin_bundler = files('blueprint/tools/bundler.py')
plugin_inc = include_directories('blueprint/include', 'blueprint/src')

subdir('blueprint/src')

plugin_library = static_library(
    'cyberether_blueprint_plugin_code',
    plugin_sources,
    include_directories: plugin_inc,
    dependencies: libjetstream_dep,
    gnu_symbol_visibility: 'hidden',
    pic: true,
    install: false,
)
plugin_code_dep = declare_dependency(
    include_directories: plugin_inc,
    dependencies: libjetstream_dep,
    link_whole: plugin_library,
)

if get_option('tests')
    subdir('blueprint/tests')
endif

plugin_module = shared_module(
    'cyberether_blueprint_plugin',
    [],
    dependencies: plugin_code_dep,
    gnu_symbol_visibility: 'hidden',
    name_prefix: '',
    build_by_default: false,
    install: false,
)

if not py.found()
    error('Python is required to bundle plugin examples.')
endif

custom_target(
    'cyberether_blueprint_plugin_cep',
    input: [plugin_module, plugin_example],
    output: 'cyberether_blueprint_plugin.cep',
    command: [
        py,
        plugin_bundler,
        '--output', '@OUTPUT@',
        '--name', 'cyberether-blueprint-plugin',
        '--version', '0.1.0',
        '--minimum-jetstream-version', meson.project_version(),
        '--target', 'path=@INPUT0@,system=' + host_machine.system() + ',device=cpu,arch=' + host_machine.cpu_family(),
        '--example', '@INPUT1@',
    ],
    build_by_default: true,
    install: false,
)

sum_lst += {'Plugin blueprint': true}
