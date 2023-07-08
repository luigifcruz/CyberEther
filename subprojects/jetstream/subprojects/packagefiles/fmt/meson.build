project('fmt', 'cpp', version: '9.1.0', default_options: ['cpp_std=c++14'])

fmt_private_cpp_args = []
fmt_interface_cpp_args = []
fmt_private_cpp_args += ['-DFMT_EXPORT']
fmt_interface_cpp_args += ['-DFMT_SHARED']

fmt_lib = library(
  'fmt',
  'src/format.cc',
  'src/os.cc',
  cpp_args: fmt_private_cpp_args,
  include_directories: 'include',
  install: true,
)

fmt_dep = declare_dependency(
  include_directories: 'include',
  compile_args: fmt_interface_cpp_args,
  link_with: fmt_lib,
)

install_subdir('include/fmt', install_dir: get_option('includedir'))