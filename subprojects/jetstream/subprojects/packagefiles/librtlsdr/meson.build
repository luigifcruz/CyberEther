project('librtlsdr', 'c', version: '0.6.0', default_options: ['cpp_std=c++11'])

private_cpp_args = []
interface_cpp_args = []
private_cpp_args += []
interface_cpp_args += []
dep_lst = [
  dependency('libusb', required: true),
]

librtlsdr_lib = library(
  'librtlsdr',
  'src/librtlsdr.c',
  'src/tuner_e4k.c',
  'src/tuner_fc0012.c',
  'src/tuner_fc0013.c',
  'src/tuner_fc2580.c',
  'src/tuner_r82xx.c',
  'src/convenience/convenience.c',
  'src/getopt/getopt.c',
  cpp_args: private_cpp_args,
  include_directories: ['include'],
  dependencies: dep_lst,
  install: true,
)

librtlsdr_dep = declare_dependency(
  include_directories: ['include'],
  compile_args: interface_cpp_args,
  link_with: librtlsdr_lib,
  dependencies: dep_lst,
)

install_subdir('include/librtlsdr', install_dir: get_option('includedir'))
