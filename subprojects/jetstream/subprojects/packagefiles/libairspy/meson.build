project('libairspy', ['c', 'cpp'], version: '1.0.10', default_options: ['cpp_std=c++11'])

private_cpp_args = []
interface_cpp_args = []
private_cpp_args += []
interface_cpp_args += []
dep_lst = [
  dependency('libusb', required: true),
]

libairspy_lib = library(
  'libairspy',
  'libairspy/src/airspy.c',
  'libairspy/src/iqconverter_float.c',
  'libairspy/src/iqconverter_int16.c',
  cpp_args: private_cpp_args,
  include_directories: ['libairspy/src'],
  dependencies: dep_lst,
  install: true,
)

libairspy_dep = declare_dependency(
  include_directories: ['libairspy/src'],
  compile_args: interface_cpp_args,
  link_with: libairspy_lib,
  dependencies: dep_lst,
)

install_subdir('include/libairspy', install_dir: get_option('includedir'))
