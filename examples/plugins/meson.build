plugin_sources = []
plugin_example = files('blueprint/examples/blueprint_gain.yml')
plugin_bundler = files('blueprint/tools/bundler.py')

subdir('blueprint/src')

plugin_module = shared_module(
    'cyberether_blueprint_plugin',
    plugin_sources,
    include_directories: include_directories('blueprint/include', 'blueprint/src'),
    dependencies: libjetstream_dep,
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
