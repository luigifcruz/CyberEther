project('libusb', 'cpp', version: '0.3.3', default_options: ['cpp_std=c++11'])

private_cpp_args = []
interface_cpp_args = []
private_cpp_args += ['']
interface_cpp_args += []

deps = [
  dependency('threads', required: true),
]

libusb_lib = library(
  'libusb',
  'src/interface.cc',
  'src/libusb.cc',
  cpp_args: private_cpp_args,
  dependencies: deps,
  include_directories: ['include'],
  install: true,
)

libusb_dep = declare_dependency(
  include_directories: ['include'],
  compile_args: interface_cpp_args,
  dependencies: deps,
  link_with: libusb_lib,
)

meson.override_dependency('libusb', libusb_dep)