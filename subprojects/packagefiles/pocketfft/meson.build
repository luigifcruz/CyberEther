project('pocketfft', 'cpp',
    version: '5fb931a048d4ed6e2be304bbfb608f655f3da6bf',
    default_options: ['cpp_std=c++11'],
)

pocketfft_dep = declare_dependency(include_directories: include_directories('.'))
meson.override_dependency('pocketfft', pocketfft_dep)
