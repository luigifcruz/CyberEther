project('limesuiteng', ['c', 'cpp'], version: '25.1.0', default_options: ['c_std=gnu17,c17', 'cpp_std=c++17', 'optimization=2'])

private_c_args = []
private_cpp_args = []
usb_transport_cpp_args = []
cpp = meson.get_compiler('cpp')
system = host_machine.system()

if system.startswith('windows')
  private_c_args += ['-D_USE_MATH_DEFINES']
  private_cpp_args += ['-D_CRT_SECURE_NO_WARNINGS', '-DNOMINMAX', '-D_USE_MATH_DEFINES']
  usb_transport_cpp_args += ['-D__unix__=1']

  if cpp.get_id() == 'msvc' or cpp.get_id() == 'clang-cl'
    private_cpp_args += ['/FIchrono']
  endif
elif system == 'darwin' or system == 'emscripten'
  # Force the libusb-backed transport paths for cross-platform static builds.
  private_c_args += ['-D__unix__=1']
  private_cpp_args += ['-D__unix__=1']
endif

pcie_enabled = system == 'linux'

thread_dep = dependency('threads')

if system == 'emscripten'
  usb_dep = subproject(
    'libusb-browser',
    required: true,
    default_options: [
      'default_library=static',
    ],
  ).get_variable('libusb_dep')
else
  usb_dep = subproject(
    'libusb',
    required: true,
    default_options: [
      'default_library=static',
      'examples=false',
      'tests=false',
    ],
  ).get_variable('libusb_dep')
endif

atomic_code = '''
#include <atomic>
#include <cstdint>
int main() {
  std::atomic<std::uint64_t> value{0};
  value.fetch_add(1);
  return static_cast<int>(value.load());
}
'''

limesuiteng_deps = [thread_dep, usb_dep]
if not cpp.links(atomic_code, name: 'std::atomic without libatomic')
  limesuiteng_deps += [cpp.find_library('atomic', required: true)]
endif

if system.startswith('windows')
  limesuiteng_deps += [cpp.find_library('shell32', required: true)]
endif

py = import('python').find_installation(required: true)
external_sources = custom_target(
  'limesuiteng_external',
  input: files(
    'external/kissfft.zip',
    'external/rang.zip',
    'external/cpp-feather-ini-parser.zip',
  ),
  output: [
    'kiss_fft.c',
    'kiss_fft.h',
    '_kiss_fft_guts.h',
    'kiss_fft_log.h',
    'rang.hpp',
    'INI.h',
  ],
  command: [
    py,
    files('extract.py'),
    '@OUTDIR@',
    '@INPUT@',
  ],
)

version_conf = configuration_data()
version_conf.set('PROJECT_VERSION', '0.3.0')
version_conf.set('LIMESUITENG_EXTVER', 'v' + meson.project_version())
version_conf.set('LIMESUITENG_SOVER', '0.3-1')
version_conf.set('BUILD_TIMESTAMP', 'unknown')
version_conf.set('LIME_SUITE_ROOT', get_option('prefix').replace('\\', '\\\\'))
version_conf.set('VERSION_MAJOR', '0')
version_conf.set('VERSION_MINOR', '3')

version_info = configure_file(
  input: 'src/include/limesuiteng/VersionInfo.cpp.in',
  output: 'VersionInfo.cpp',
  configuration: version_conf,
)

system_resources = configure_file(
  input: 'src/SystemResources.in.cpp',
  output: 'SystemResources.cpp',
  configuration: version_conf,
)

boards_conf = configuration_data()
boards_conf.set10('LIMESUITENG_USB_FX3', true)
boards_conf.set10('LIMESUITENG_USB_FTDI', true)
boards_conf.set10('LIMESUITENG_PCIE', pcie_enabled)

board_support = configure_file(
  input: 'src/boards/BoardSupportLoader.in.cpp',
  output: 'BoardSupportLoader.cpp',
  configuration: boards_conf,
  format: 'cmake',
)

subdir('generated')

limesuiteng_public_inc = include_directories(
  'src/include',
  'generated',
)

limesuiteng_inc = include_directories(
  '.',
  'src',
  'src/threadHelper',
  'embedded',
  'embedded/include',
)

limesuiteng_c_sources = files(
  'embedded/gfir/corrections.c',
  'embedded/gfir/gfir_lms.c',
  'embedded/gfir/lms.c',
  'embedded/gfir/recipes.c',
  'embedded/gfir/rounding.c',
  'embedded/lms7002m/calibrations.c',
  'embedded/lms7002m/lms7002m.c',
  'embedded/lms7002m/privates.c',
  'embedded/lms7002m/save_state.c',
  'embedded/lms7002m/spi.c',
)

limesuiteng_sources = files(
  'src/CommonFunctions.cpp',
  'src/OEMTesting.cpp',
  'src/logger/Logger.cpp',
  'src/logger/LoggerCString.cpp',
  'src/protocols/LMS64CProtocol.cpp',
  'src/protocols/LMSBoards.cpp',
  'src/FPGA/FPGA_common.cpp',
  'src/FPGA/WriteRegistersBatch.cpp',
  'src/threadHelper/threadHelper.cpp',
  'src/memory/MemoryPool.cpp',
  'src/API/LimePlugin.cpp',
  'src/include/limesuiteng/SDRDevice.cpp',
  'src/utilities/toString.cpp',
  'src/comms/SPIChipSelectShim.cpp',
  'src/comms/SPI_utilities.cpp',
  'src/boards/DeviceRegistry.cpp',
  'src/boards/DeviceHandle.cpp',
  'src/boards/LMS7002M_SDRDevice.cpp',
  'src/chips/ADF4002/ADF4002.cpp',
  'src/chips/CDCM6208/CDCM6208.cpp',
  'src/chips/LMS7002M/LMS7002M_RegistersMap.cpp',
  'src/chips/LMS7002M/LMS7002M.cpp',
  'src/chips/LMS7002M/LMS7002MCSR_Data.cpp',
  'src/chips/LMS7002M/LMS7002MCSR_To_LMS7002MCSR_Data.cpp',
  'src/chips/LMS7002M/validation.cpp',
  'src/chips/LMS7002M/filtersCalibration.cpp',
  'src/chips/LMS7002M/gainCalibrations.cpp',
  'src/chips/LMS7002M/mcu_dc_iq_calibration.cpp',
  'src/chips/LMS7002M/MCU_BD.cpp',
  'src/chips/LMS7002M/MCU_File.cpp',
  'src/chips/Si5351C/Si5351C.cpp',
  'src/DSP/CFR/CrestFactorReduction.cpp',
  'src/DSP/FFT/FFT.cpp',
  'src/streaming/samplesConversion.cpp',
  'src/streaming/TRXLooper.cpp',
  'src/streaming/BufferInterleaving.cpp',
  'src/streaming/AvgRmsCounter.cpp',
  'src/streaming/StreamComposite.cpp',
  'src/comms/USB/LMS64C_FPGA_Over_USB.cpp',
  'src/comms/USB/LMS64C_LMS7002M_Over_USB.cpp',
  'src/comms/USB/LMS64C_ADF4002_Over_USB.cpp',
  'src/comms/USB/USBDMAEmulation.cpp',
)

limesuiteng_usb_sources = files(
  'src/comms/USB/UnixUsb.cpp',
  'src/comms/USB/FX3/FX3.cpp',
  'src/comms/USB/FT601/FT601.cpp',
  'src/boards/LimeSDR/LimeSDR.cpp',
  'src/boards/LimeSDR/USB_CSR_Pipe_SDR.cpp',
  'src/boards/LimeSDR/LMS64C_ADF_Over_USB.cpp',
  'src/boards/DeviceFactoryFX3.cpp',
  'src/boards/LimeSDR_Mini/LimeSDR_Mini.cpp',
  'src/boards/LimeSDR_Mini/FPGA_Mini.cpp',
  'src/boards/LimeSDR_Mini/USB_CSR_Pipe_Mini.cpp',
  'src/boards/LimeSDR_Mini/LimeNET_Micro.cpp',
  'src/boards/DeviceFactoryFTDI.cpp',
)

if pcie_enabled
  limesuiteng_sources += files(
    'src/comms/PCIe/LMS64C_FPGA_Over_PCIe.cpp',
    'src/comms/PCIe/LMS64C_LMS7002M_Over_PCIe.cpp',
    'src/comms/PCIe/LimePCIe.cpp',
    'src/comms/PCIe/LimePCIeDMA.cpp',
    'src/comms/PCIe/PCIE_CSR_Pipe.cpp',
    'src/boards/DeviceFactoryPCIe.cpp',
    'src/boards/LimeSDR_X3/FPGA_X3.cpp',
    'src/boards/LimeSDR_X3/LimeSDR_X3.cpp',
    'src/boards/LimeSDR_X3/SlaveSelectShim.cpp',
    'src/boards/LimeSDR_XTRX/LimeSDR_XTRX.cpp',
    'src/boards/LimeSDR_XTRX/FPGA_XTRX.cpp',
    'src/boards/MMX8/LMS64C_ADF_Over_PCIe_MMX8.cpp',
    'src/boards/MMX8/LMS64C_FPGA_Over_PCIe_MMX8.cpp',
    'src/boards/MMX8/LMS64C_LMS7002M_Over_PCIe_MMX8.cpp',
    'src/boards/MMX8/MM_X8.cpp',
    'src/boards/MMX8/RFStream_X8.cpp',
  )
endif

limesuiteng_extra_objects = []
if system.startswith('windows')
  limesuiteng_usb_lib = static_library(
    'limesuiteng_usb_connections',
    limesuiteng_usb_sources,
    cpp_args: private_cpp_args + usb_transport_cpp_args,
    include_directories: [limesuiteng_inc, limesuiteng_public_inc],
    dependencies: limesuiteng_deps,
  )
  limesuiteng_extra_objects += limesuiteng_usb_lib.extract_all_objects()
else
  limesuiteng_sources += limesuiteng_usb_sources
endif

limesuiteng_c_lib = static_library(
  'limesuiteng_embedded',
  limesuiteng_c_sources,
  c_args: private_c_args,
  include_directories: limesuiteng_inc,
)

limesuiteng_lib = library(
  'limesuiteng',
  limesuiteng_sources,
  external_sources,
  version_info,
  system_resources,
  board_support,
  c_args: private_c_args,
  cpp_args: private_cpp_args,
  include_directories: [limesuiteng_inc, limesuiteng_public_inc],
  dependencies: limesuiteng_deps,
  link_whole: limesuiteng_c_lib,
  objects: limesuiteng_extra_objects,
  install: true,
)

limesuiteng_dep = declare_dependency(
  include_directories: limesuiteng_public_inc,
  dependencies: limesuiteng_deps,
  link_with: limesuiteng_lib,
)
