project(
    'cyberether-blueprint-plugin',
    ['cpp'],
    version: '0.1.0',
    meson_version: '>=1.11.0',
    default_options: [
        'cpp_std=c++20',
        'buildtype=release',
        'b_ndebug=if-release',
        'warning_level=3',
    ]
)

plugin_id = 'blueprint'
plugin_title = 'Blueprint'
plugin_summary = 'A minimal gain block to start new plugins from.'
plugin_author = 'CyberEther'
plugin_license = 'MIT'

cyberether_dep = dependency(
    'jetstream',
    required: false,
    fallback: ['cyberether', 'libjetstream_dep'],
    default_options: [
        'tests=false',
        'examples=false',
        'remote=disabled',
    ],
)

if not cyberether_dep.found()
    error('CyberEther dependency not found.')
endif

cyberether_version = cyberether_dep.version()
cyberether_build_id = cyberether_dep.get_variable('build_id', default_value: '')
cyberether_backends = cyberether_dep.get_variable('backends', default_value: '').split(',')

if cyberether_build_id == ''
    error('This CyberEther SDK does not export a build ID. Build against a newer CyberEther.')
endif

plugin_inc = include_directories('include', 'src')
plugin_sources = []
plugin_test_lst = []
plugin_examples = files('examples/gain.yml')
plugin_bundler = files('tools/bundler.py')

plugin_device_cpu_enabled = 'cpu' in cyberether_backends
if not plugin_device_cpu_enabled
    error('This CyberEther build has no backend that the blueprint implements.')
endif

pymod = import('python')
py = pymod.find_installation(required: true)

subdir('src')

plugin_library = static_library(
    'cyberether_blueprint_plugin_code',
    plugin_sources,
    include_directories: plugin_inc,
    dependencies: cyberether_dep,
    gnu_symbol_visibility: 'hidden',
    pic: true,
    install: false,
)
plugin_code_dep = declare_dependency(
    include_directories: plugin_inc,
    dependencies: cyberether_dep,
    link_whole: plugin_library,
)

if get_option('tests')
    subdir('tests')
endif

plugin_module = shared_module(
    plugin_id,
    [],
    dependencies: plugin_code_dep,
    gnu_symbol_visibility: 'hidden',
    name_prefix: '',
    build_by_default: false,
    install: false,
)

plugin_bundle_args = [
    py,
    plugin_bundler,
    '--output', '@OUTPUT@',
    '--id', plugin_id,
    '--title', plugin_title,
    '--summary', plugin_summary,
    '--author', plugin_author,
    '--license', plugin_license,
    '--version', meson.project_version(),
    '--cyberether', cyberether_version,
    '--build', cyberether_build_id,
    '--library', '@INPUT0@',
]
foreach example : plugin_examples
    plugin_bundle_args += ['--example', example]
endforeach

custom_target(
    'cyberether_blueprint_plugin_cep',
    input: [plugin_module],
    depend_files: plugin_examples,
    output: '-'.join(plugin_id, meson.project_version(), 'cyberether', cyberether_version, cyberether_build_id) + '.cep',
    command: plugin_bundle_args,
    build_by_default: true,
    install: false,
)

summary({
    'Build ID': cyberether_build_id,
    'Backends': ','.join(cyberether_backends),
    'CPU': plugin_device_cpu_enabled,
}, section: 'CyberEther', bool_yn: true)
