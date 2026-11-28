project('libdatachannel', 'cpp', version: '0.24.8', license: 'MPL-2.0', default_options: ['cpp_std=c++17', 'optimization=2'])

cxx = meson.get_compiler('cpp')
is_windows = host_machine.system() == 'windows'

deps = [
  dependency('threads'),
  subproject('plog').get_variable('plog_dep'),
  subproject('libjuice').get_variable('libjuice_dep'),
  dependency('usrsctp'),
  dependency('libsrtp2'),
  dependency('openssl'),
]
if is_windows
  deps += [cxx.find_library('ws2_32'), cxx.find_library('bcrypt')]
endif

public_args = [
  '-DRTC_STATIC',
  '-DRTC_ENABLE_WEBSOCKET=0',
  '-DRTC_ENABLE_MEDIA=1',
]

private_args = public_args + [
  '-DRTC_EXPORTS',
  '-DRTC_SYSTEM_SRTP=0',
  '-DRTC_SYSTEM_JUICE=0',
  '-DUSE_GNUTLS=0',
  '-DUSE_MBEDTLS=0',
  '-DUSE_NICE=0',
]
if is_windows
  private_args += ['-DWIN32_LEAN_AND_MEAN', '-DNOMINMAX', '-D_CRT_SECURE_NO_WARNINGS']
endif

sources = files(
  'src/av1rtppacketizer.cpp',
  'src/candidate.cpp',
  'src/capi.cpp',
  'src/channel.cpp',
  'src/configuration.cpp',
  'src/datachannel.cpp',
  'src/dependencydescriptor.cpp',
  'src/description.cpp',
  'src/global.cpp',
  'src/h264rtpdepacketizer.cpp',
  'src/h264rtppacketizer.cpp',
  'src/h265nalunit.cpp',
  'src/h265rtpdepacketizer.cpp',
  'src/h265rtppacketizer.cpp',
  'src/iceudpmuxlistener.cpp',
  'src/impl/certificate.cpp',
  'src/impl/channel.cpp',
  'src/impl/datachannel.cpp',
  'src/impl/dtlssrtptransport.cpp',
  'src/impl/dtlstransport.cpp',
  'src/impl/http.cpp',
  'src/impl/httpproxytransport.cpp',
  'src/impl/icetransport.cpp',
  'src/impl/iceudpmuxlistener.cpp',
  'src/impl/init.cpp',
  'src/impl/logcounter.cpp',
  'src/impl/peerconnection.cpp',
  'src/impl/pollinterrupter.cpp',
  'src/impl/pollservice.cpp',
  'src/impl/processor.cpp',
  'src/impl/sctptransport.cpp',
  'src/impl/sha.cpp',
  'src/impl/tcpserver.cpp',
  'src/impl/tcptransport.cpp',
  'src/impl/threadpool.cpp',
  'src/impl/tls.cpp',
  'src/impl/tlstransport.cpp',
  'src/impl/track.cpp',
  'src/impl/transport.cpp',
  'src/impl/utils.cpp',
  'src/impl/verifiedtlstransport.cpp',
  'src/impl/websocket.cpp',
  'src/impl/websocketserver.cpp',
  'src/impl/wshandshake.cpp',
  'src/impl/wstransport.cpp',
  'src/mediahandler.cpp',
  'src/message.cpp',
  'src/nalunit.cpp',
  'src/pacinghandler.cpp',
  'src/peerconnection.cpp',
  'src/plihandler.cpp',
  'src/rembhandler.cpp',
  'src/rtcpnackresponder.cpp',
  'src/rtcpreceivingsession.cpp',
  'src/rtcpsrreporter.cpp',
  'src/rtp.cpp',
  'src/rtpdepacketizer.cpp',
  'src/rtppacketizationconfig.cpp',
  'src/rtppacketizer.cpp',
  'src/track.cpp',
  'src/websocket.cpp',
  'src/websocketserver.cpp',
)

libdatachannel = static_library('datachannel', sources,
  cpp_args: private_args + cxx.get_supported_arguments(['-Wno-unused-parameter', '-Wno-deprecated-declarations']),
  include_directories: include_directories('include', 'include/rtc', 'src'),
  dependencies: deps,
  pic: true,
)

libdatachannel_dep = declare_dependency(
  compile_args: public_args,
  include_directories: include_directories('include'),
  link_with: libdatachannel,
  dependencies: deps,
)
meson.override_dependency('libdatachannel', libdatachannel_dep)
