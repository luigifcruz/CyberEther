project('soapyremote', 'cpp', version: '0.8.1', license: 'BSD', default_options: ['cpp_std=c++11'])

private_cpp_args = []
interface_cpp_args = []
private_cpp_args += ['-DSTRERROR_R_XSI']
interface_cpp_args += []

soapyremote_dep = [
  dependency('soapysdr')
]

soapyremote_lib = library(
  'SoapyRemote',
  'client/Streaming.cpp',
  'client/ClientStreamData.cpp',
  'client/DiscoverServers.cpp',
  'client/LogAcceptor.cpp',
  'client/Registration.cpp',
  'client/Settings.cpp',
  'common/SoapyRPCPacker.cpp',
  'common/SoapyHTTPUtils.cpp',
  'common/SoapyIfAddrs.cpp',
  'common/SoapyInfoUtils.in.cpp',
  'common/SoapyMDNSEndpointApple.cpp',
  'common/SoapyRPCSocket.cpp',
  'common/SoapyRPCUnpacker.cpp',
  'common/SoapySSDPEndpoint.cpp',
  'common/SoapyStreamEndpoint.cpp',
  'common/SoapyURLUtils.cpp',
  cpp_args: private_cpp_args,
  dependencies: soapyremote_dep,
  include_directories: 'common',
  install: true,
)

soapyremote_dep = declare_dependency(
  include_directories: 'common',
  compile_args: interface_cpp_args,
  dependencies: soapyremote_dep,
  link_with: soapyremote_lib,
)
