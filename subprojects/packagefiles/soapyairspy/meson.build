project('soapyairspy', 'cpp', version: '0.8.1', default_options: ['cpp_std=c++11', 'optimization=2'])

private_cpp_args = []
interface_cpp_args = []
private_cpp_args += ['-DSTRERROR_R_XSI']
interface_cpp_args += []

deps = [
  dependency('soapysdr', required: true),
  dependency('libairspy', required: true),
]

soapyairspy_lib = library(
  'SoapyAirspy',
  'Registration.cpp',
  'Settings.cpp',
  'Streaming.cpp',
  cpp_args: private_cpp_args,
  dependencies: deps,
  include_directories: ['.'],
  install: true,
)

soapyairspy_dep = declare_dependency(
  include_directories: ['.'],
  compile_args: interface_cpp_args,
  dependencies: deps,
  link_with: soapyairspy_lib,
)
