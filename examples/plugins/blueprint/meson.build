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

jst_is_browser = host_machine.system().startswith('emscripten')

cyberether_dep = dependency(
    'jetstream',
    required: false,
    fallback: ['cyberether', 'libjetstream_dep'],
    default_options: [
        'tests=false',
        'examples=false',
    ],
)

if not cyberether_dep.found()
    error('CyberEther dependency not found.')
endif

if jst_is_browser
    cyberether_dep = cyberether_dep.partial_dependency(
        compile_args: true,
        includes: true,
    )
endif

plugin_inc = include_directories('include', 'src')
plugin_sources = []
plugin_test_lst = []
plugin_example = files('examples/blueprint_gain.yml')
plugin_bundler = files('tools/bundler.py')

pymod = import('python')
py = pymod.find_installation(required: true)

subdir('src')

plugin_library = static_library(
    'cyberether_blueprint_plugin_code',
    plugin_sources,
    include_directories: plugin_inc,
    dependencies: cyberether_dep,
    gnu_symbol_visibility: jst_is_browser ? 'default' : 'hidden',
    pic: true,
    install: false,
)
plugin_code_dep = declare_dependency(
    include_directories: plugin_inc,
    dependencies: cyberether_dep,
    link_whole: plugin_library,
)

if get_option('tests') and not jst_is_browser
    subdir('tests')
endif

if jst_is_browser
    plugin_module = shared_module(
        'cyberether_blueprint_plugin',
        [],
        dependencies: plugin_code_dep,
        link_args: meson.get_external_property('side_module_link_args'),
        gnu_symbol_visibility: 'default',
        name_prefix: '',
        name_suffix: 'wasm',
        build_by_default: false,
        install: false,
    )
else
    plugin_module = shared_module(
        'cyberether_blueprint_plugin',
        [],
        dependencies: plugin_code_dep,
        gnu_symbol_visibility: 'hidden',
        name_prefix: '',
        build_by_default: false,
        install: false,
    )
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
        '--version', meson.project_version(),
        '--minimum-jetstream-version', '1.7.0',
        '--target', 'path=@INPUT0@,system=' + host_machine.system() + ',device=cpu,arch=' + host_machine.cpu_family(),
        '--example', '@INPUT1@',
    ],
    build_by_default: true,
    install: false,
)
