project('soapybladerf', 'cpp', version: '0.4.2', default_options: ['cpp_std=c++11', 'optimization=2'])

deps = [
  dependency('soapysdr', required: true),
  subproject(
    'libbladerf',
    required: true,
    default_options: ['default_library=static'],
  ).get_variable('libbladerf_dep'),
]

private_cpp_args = []
if target_machine.system().startswith('windows')
  private_cpp_args += ['-DNOMINMAX']
endif

soapybladerf_lib = library(
  'SoapyBladeRF',
  'bladeRF_Registration.cpp',
  'bladeRF_Settings.cpp',
  'bladeRF_Streaming.cpp',
  cpp_args: private_cpp_args,
  dependencies: deps,
  include_directories: ['.'],
  install: true,
)

soapybladerf_dep = declare_dependency(
  include_directories: ['.'],
  dependencies: deps,
  link_with: soapybladerf_lib,
)
