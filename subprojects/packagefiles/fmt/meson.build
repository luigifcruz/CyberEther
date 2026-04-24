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

fmt_dep = declare_dependency(
  include_directories: 'include',
  compile_args: '-DJST_FMT_HEADER_ONLY',
)

install_subdir(
  'include/jetstream/fmt',
  install_dir: get_option('includedir') / 'jetstream',
)
