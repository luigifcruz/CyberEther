project(
    'cyberether-blueprint-plugin',
    ['cpp'],
    version: '0.1.0',
    meson_version: '>=1.11.0',
    default_options: [
        'cpp_std=c++20',
        'warning_level=3',
    ]
)

cyberether_dep = dependency(
    'cyberether',
    required: false,
    fallback: ['cyberether', 'libjetstream_dep'],
)

if not cyberether_dep.found()
    error('CyberEther dependency not found.')
endif

plugin_inc = include_directories('include', 'src')
plugin_sources = []
plugin_example = files('examples/blueprint_gain.yml')
plugin_bundler = files('tools/bundler')

pymod = import('python')
py = pymod.find_installation(required: true)

subdir('src')

plugin_module = shared_module(
    'cyberether_blueprint_plugin',
    plugin_sources,
    include_directories: plugin_inc,
    dependencies: cyberether_dep,
    gnu_symbol_visibility: 'hidden',
    name_prefix: '',
    build_by_default: false,
    install: false,
)

custom_target(
    'cyberether_blueprint_plugin_cep',
    input: [plugin_module, plugin_example],
    output: 'cyberether_blueprint_plugin.cep',
    command: [
        py,
        plugin_bundler,
        '--output', '@OUTPUT@',
        '--name', 'cyberether-blueprint-plugin',
        '--version', meson.project_version(),
        '--minimum-jetstream-version', '1.5.0',
        '--target', 'path=@INPUT0@,system=' + host_machine.system() + ',device=cpu,arch=' + host_machine.cpu_family(),
        '--example', '@INPUT1@',
    ],
    build_by_default: true,
    install: false,
)
