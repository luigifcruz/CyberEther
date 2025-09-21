project(
  'fmt',
  'cpp',
  version: '11.2.0',
  default_options: ['cpp_std=c++14'],
)

bash = find_program('bash')
script = files('patch.sh')

run_command(
  bash, script[0], meson.source_root() / 'subprojects/fmt-11.2.0/include/fmt',
  check: true
)

fmt_dep = declare_dependency(
  include_directories: 'include',
  compile_args: '-DFMT_HEADER_ONLY',
)
