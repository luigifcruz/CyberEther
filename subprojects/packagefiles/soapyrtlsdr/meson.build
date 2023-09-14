project('soapyrtlsdr', 'cpp', version: '0.3.3', default_options: ['cpp_std=c++11'])

private_cpp_args = []
interface_cpp_args = []
private_cpp_args += ['-DSTRERROR_R_XSI']
interface_cpp_args += []

deps = [
  dependency('soapysdr', required: true),
  dependency('librtlsdr', required: true),
]

soapyrtlsdr_lib = library(
  'SoapyRTLSDR',
  'Registration.cpp',
  'Settings.cpp',
  'Streaming.cpp',
  cpp_args: private_cpp_args,
  dependencies: deps,
  include_directories: ['.'],
  install: true,
)

soapyrtlsdr_dep = declare_dependency(
  include_directories: ['.'],
  compile_args: interface_cpp_args,
  dependencies: deps,
  link_with: soapyrtlsdr_lib,
)
