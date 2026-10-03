project(
  'fmt',
  'cpp',
  version: '11.2.0',
  default_options: ['cpp_std=c++14'],
)

py = import('python').find_installation(required: true)
script = files('patch.py')

run_command(
  py, script[0], meson.current_source_dir() / 'include/fmt',
  check: true
)

if meson.get_compiler('cpp').get_id() == 'msvc'
  fmt_source = configure_file(
    input: 'src/format.cc',
    output: 'jetstream-format.cc',
    command: [py, script[0], '--source', '@INPUT@', '@OUTPUT@'],
  )
  fmt_lib = static_library(
    'jetstream-fmt', fmt_source,
    include_directories: 'include',
    override_options: ['cpp_std=c++20'],
  )
  fmt_dep = declare_dependency(
    include_directories: 'include',
    compile_args: '-DJST_FMT_COMPILED',
    link_with: fmt_lib,
  )
else
  fmt_dep = declare_dependency(
    include_directories: 'include',
    compile_args: '-DJST_FMT_HEADER_ONLY',
  )
endif

install_subdir(
  'include/jetstream/fmt',
  install_dir: get_option('includedir') / 'jetstream',
)
