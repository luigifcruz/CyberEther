project('soapyremote', 'cpp', version: '0.8.1', default_options: ['cpp_std=c++11', 'optimization=2'])

private_cpp_args = []
interface_cpp_args = []
private_cpp_args += ['-DSTRERROR_R_XSI']
interface_cpp_args += []

deps = [
  dependency('soapysdr')
]

src_lst = [
  'client/Streaming.cpp',
  'client/ClientStreamData.cpp',
  'client/DiscoverServers.cpp',
  'client/LogAcceptor.cpp',
  'client/Registration.cpp',
  'client/Settings.cpp',
  'common/SoapyRPCPacker.cpp',
  'common/SoapyHTTPUtils.cpp',
  'common/SoapyIfAddrs.cpp',
  'common/SoapyIfAddrsUnix.cpp',
  'common/SoapyInfoUtils.in.cpp',
  'common/SoapyRPCSocket.cpp',
  'common/SoapyRPCUnpacker.cpp',
  'common/SoapySSDPEndpoint.cpp',
  'common/SoapyStreamEndpoint.cpp',
  'common/SoapyURLUtils.cpp',
]

if target_machine.system() == 'ios'
  src_lst += [
    'common/SoapyMDNSEndpointApple.cpp',
  ]
endif

soapyremote_lib = library(
  'SoapyRemote',
  src_lst,
  cpp_args: private_cpp_args,
  dependencies: deps,
  include_directories: ['common', 'client'],
  install: true,
)

soapyremote_dep = declare_dependency(
  include_directories: ['common', 'client'],
  compile_args: interface_cpp_args,
  dependencies: deps,
  link_with: soapyremote_lib,
)
