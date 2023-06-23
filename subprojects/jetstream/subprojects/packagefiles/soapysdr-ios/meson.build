project('soapysdr', 'cpp', version: '0.8.1', license: 'BSD', default_options: ['cpp_std=c++11'])

private_cpp_args = []
interface_cpp_args = []
private_cpp_args += []
interface_cpp_args += []

soapysdr_lib = library(
  'SoapySDR',
  'lib/ConverterRegistry.cpp',
  'lib/ConvertersC.cpp',
  'lib/DefaultConverters.cpp',
  'lib/Device.cpp',
  'lib/DeviceC.cpp',
  'lib/Errors.cpp',
  'lib/ErrorsC.cpp',
  'lib/Factory.cpp',
  'lib/FactoryC.cpp',
  'lib/Formats.cpp',
  'lib/FormatsC.cpp',
  'lib/Logger.cpp',
  'lib/LoggerC.cpp',
  'lib/Modules.in.cpp',
  'lib/ModulesC.cpp',
  'lib/NullDevice.cpp',
  'lib/Registry.cpp',
  'lib/TimeC.cpp',
  'lib/Types.cpp',
  'lib/TypesC.cpp',
  'lib/Version.in.cpp',
  'lib/VersionC.cpp',
  cpp_args: private_cpp_args,
  include_directories: 'include',
  install: true,
)

soapysdr_dep = declare_dependency(
  include_directories: 'include',
  compile_args: interface_cpp_args,
  link_with: soapysdr_lib,
)

install_subdir('include/SoapySDR', install_dir: get_option('includedir'))
