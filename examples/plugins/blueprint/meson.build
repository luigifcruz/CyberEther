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

subdir('src')

shared_module(
    'cyberether_blueprint_plugin',
    plugin_sources,
    include_directories: plugin_inc,
    dependencies: cyberether_dep,
    gnu_symbol_visibility: 'hidden',
    name_prefix: '',
    build_by_default: true,
    install: false,
)
