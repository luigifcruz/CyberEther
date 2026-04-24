project('limesuite', ['c', 'cpp'], version: '23.11.0', default_options: ['cpp_std=c++11', 'optimization=2'])

private_c_args = []
private_cpp_args = []
usb_transport_cpp_args = []
cpp = meson.get_compiler('cpp')

if target_machine.system().startswith('windows')
  private_cpp_args += ['-D_CRT_SECURE_NO_WARNINGS']
  usb_transport_cpp_args += ['-D__unix__=1']

  if cpp.get_id() == 'msvc' or cpp.get_id() == 'clang-cl'
    private_cpp_args += ['/FIchrono']
  endif
elif target_machine.system() == 'darwin' or \
   target_machine.system() == 'emscripten'
  # Force the libusb-backed transport paths for cross-platform static builds.
  private_c_args += ['-D__unix__=1']
  private_cpp_args += ['-D__unix__=1']
endif

thread_dep = dependency('threads')
soapysdr_dep = dependency('soapysdr', required: true)

if target_machine.system() == 'emscripten'
  usb_dep = dependency('libusb-browser', required: true)
else
  usb_dep = dependency('libusb-1.0', required: true)
endif

atomic_dep = cpp.find_library('atomic', required: false)

limesuite_deps = [thread_dep, usb_dep]
if atomic_dep.found()
  limesuite_deps += [atomic_dep]
endif

if target_machine.system().startswith('windows')
  limesuite_deps += [cpp.find_library('shell32', required: true)]
endif

version_conf = configuration_data()
version_conf.set('BUILD_TIMESTAMP', 'unknown')
version_conf.set('LIME_SUITE_ROOT', get_option('prefix').replace('\\', '\\\\'))
version_conf.set('LIME_SUITE_SOVER', '23.11-1')
version_conf.set('LIME_SUITE_VERSION', meson.project_version())
version_conf.set('VERSION_MAJOR', '23')
version_conf.set('VERSION_MINOR', '11')

connections_conf = configuration_data()
connections_conf.set('ENABLE_EVB7COM', false)
connections_conf.set('ENABLE_FX3', true)
connections_conf.set('ENABLE_STREAM_UNITE', false)
connections_conf.set('ENABLE_FTDI', true)
connections_conf.set('ENABLE_PCIE_XILLYBUS', false)
connections_conf.set('ENABLE_REMOTE', false)
connections_conf.set('ENABLE_SPI', false)

builtin_connections = configure_file(
  input: 'BuiltinConnections.cpp.in',
  output: 'BuiltinConnections.cpp',
  configuration: connections_conf,
)

version_info = configure_file(
  input: 'VersionInfo.cpp.in',
  output: 'VersionInfo.cpp',
  configuration: version_conf,
)

system_resources = configure_file(
  input: 'SystemResources.cpp.in',
  output: 'SystemResources.cpp',
  configuration: version_conf,
)

limesuite_inc = include_directories(
  'src',
  'src/API',
  'src/lime',
  'src/GFIR',
  'src/protocols',
  'src/ConnectionRegistry',
  'src/lms7002m_mcu',
  'src/ADF4002',
  'src/Si5351C',
  'src/lms7002m',
  'src/FPGA_common',
  'src/limeRFE',
  'src/threadHelper',
  'src/ConnectionFX3',
  'src/ConnectionFTDI',
  'SoapyLMS7',
  'external/cpp-feather-ini-parser',
  'external/kissFFT',
)

limesuite_sources = files(
  'src/Logger.cpp',
  'src/ADF4002/ADF4002.cpp',
  'src/lms7002m_mcu/MCU_BD.cpp',
  'src/lms7002m_mcu/MCU_File.cpp',
  'src/ConnectionRegistry/IConnection.cpp',
  'src/ConnectionRegistry/ConnectionHandle.cpp',
  'src/ConnectionRegistry/ConnectionRegistry.cpp',
  'src/lms7002m/LMS7002M_RegistersMap.cpp',
  'src/lms7002m/LMS7002M_parameters.cpp',
  'src/lms7002m/LMS7002M.cpp',
  'src/lms7002m/LMS7002M_RxTxCalibrations.cpp',
  'src/lms7002m/LMS7002M_BaseCalibrations.cpp',
  'src/lms7002m/mcu_dc_iq_calibration.cpp',
  'src/lms7002m/LMS7002M_filtersCalibration.cpp',
  'src/lms7002m/LMS7002M_gainCalibrations.cpp',
  'src/protocols/LMS64CProtocol.cpp',
  'src/protocols/Streamer.cpp',
  'src/protocols/ConnectionImages.cpp',
  'src/Si5351C/Si5351C.cpp',
  'external/kissFFT/kiss_fft.c',
  'src/API/lms7_api.cpp',
  'src/API/lms7_device.cpp',
  'src/API/LmsGeneric.cpp',
  'src/API/qLimeSDR.cpp',
  'src/API/LimeSDR_mini.cpp',
  'src/API/LimeSDR.cpp',
  'src/API/LimeNET_micro.cpp',
  'src/API/LimeSDR_PCIE.cpp',
  'src/API/LimeSDR_Core.cpp',
  'src/FPGA_common/FPGA_common.cpp',
  'src/FPGA_common/FPGA_Mini.cpp',
  'src/FPGA_common/FPGA_Q.cpp',
  'src/windowFunction.cpp',
  'src/threadHelper/threadHelper.cpp',
  'src/GFIR/corrections.c',
  'src/GFIR/gfir_lms.c',
  'src/GFIR/lms.c',
  'src/GFIR/recipes.c',
  'src/GFIR/rounding.c',
)

limesuite_usb_sources = files(
  'src/ConnectionFX3/ConnectionFX3Entry.cpp',
  'src/ConnectionFX3/ConnectionFX3.cpp',
  'src/ConnectionFTDI/ConnectionFT601Entry.cpp',
  'src/ConnectionFTDI/ConnectionFT601.cpp',
)

limesuite_extra_objects = []
if target_machine.system().startswith('windows')
  limesuite_usb_lib = static_library(
    'limesuite_usb_connections',
    limesuite_usb_sources,
    cpp_args: private_cpp_args + usb_transport_cpp_args,
    include_directories: limesuite_inc,
    dependencies: limesuite_deps,
  )
  limesuite_extra_objects += limesuite_usb_lib.extract_all_objects()
else
  limesuite_sources += limesuite_usb_sources
endif

limesuite_lib = library(
  'LimeSuite',
  limesuite_sources,
  builtin_connections,
  version_info,
  system_resources,
  c_args: private_c_args,
  cpp_args: private_cpp_args,
  include_directories: limesuite_inc,
  dependencies: limesuite_deps,
  objects: limesuite_extra_objects,
  install: true,
)

limesuite_dep = declare_dependency(
  include_directories: limesuite_inc,
  dependencies: limesuite_deps,
  link_with: limesuite_lib,
)

soapylms7_lib = library(
  'SoapyLMS7',
  'SoapyLMS7/Registration.cpp',
  'SoapyLMS7/Settings.cpp',
  'SoapyLMS7/Streaming.cpp',
  c_args: private_c_args,
  cpp_args: private_cpp_args,
  include_directories: limesuite_inc,
  dependencies: [soapysdr_dep, limesuite_dep],
  install: true,
)

soapylms7_dep = declare_dependency(
  include_directories: limesuite_inc,
  dependencies: [soapysdr_dep, limesuite_dep],
  link_with: soapylms7_lib,
)
