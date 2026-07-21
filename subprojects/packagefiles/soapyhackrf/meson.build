project('soapyhackrf', 'cpp', version: '0.3.4', default_options: ['cpp_std=c++11', 'optimization=2'])

deps = [
  dependency('soapysdr', required: true),
  subproject(
    'libhackrf',
    required: true,
    default_options: [
      'default_library=static',
    ],
  ).get_variable('libhackrf_dep'),
]

soapyhackrf_lib = library(
  'SoapyHackRF',
  'HackRF_Registration.cpp',
  'HackRF_Settings.cpp',
  'HackRF_Streaming.cpp',
  'HackRF_Session.cpp',
  dependencies: deps,
  include_directories: ['.'],
  install: true,
)

soapyhackrf_dep = declare_dependency(
  include_directories: ['.'],
  dependencies: deps,
  link_with: soapyhackrf_lib,
)
