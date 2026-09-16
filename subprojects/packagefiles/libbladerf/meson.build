project(
  'libbladerf',
  'c',
  version: '2.6.0',
  default_options: ['c_std=gnu11,c11', 'optimization=2', 'default_library=static'],
)

cc = meson.get_compiler('c')
system = target_machine.system()
is_browser = system == 'emscripten'
is_windows = system.startswith('windows')

if is_browser
  usb_dep = dependency('libusb-browser', required: true)
else
  usb_dep = subproject(
    'libusb',
    required: true,
    default_options: ['default_library=static', 'examples=false', 'tests=false'],
  ).get_variable('libusb_dep')
endif

deps = [usb_dep, dependency('threads')]
math_dep = cc.find_library('m', required: false)
if math_dep.found()
  deps += [math_dep]
endif

host_conf = configuration_data()
# Emscripten provides the POSIX/endian interfaces used by the Linux host path.
# USB access still goes exclusively through the WebUSB libusb backend.
host_conf.set10('BLADERF_OS_LINUX', system == 'linux' or is_browser)
host_conf.set10('BLADERF_OS_FREEBSD', system == 'freebsd')
host_conf.set10('BLADERF_OS_OSX', system == 'darwin')
host_conf.set10('BLADERF_OS_WINDOWS', is_windows)
host_conf.set10('BLADERF_BIG_ENDIAN', target_machine.endian() == 'big')
have_clock_gettime = cc.has_function('clock_gettime', prefix: '#include <time.h>')
host_conf.set10('HAVE_CLOCK_GETTIME', have_clock_gettime)
configure_file(
  input: 'host/common/include/host_config.h.in',
  output: 'host_config.h',
  configuration: host_conf,
  format: 'cmake',
)

version_conf = configuration_data()
version_conf.set('VERSION', meson.project_version())
version_conf.set('VERSION_INFO_MAJOR', 2)
version_conf.set('VERSION_INFO_MINOR', 6)
version_conf.set('VERSION_INFO_PATCH', 0)
configure_file(
  input: 'host/libraries/libbladeRF/src/version.h.in',
  output: 'version.h',
  configuration: version_conf,
)

backend_conf = configuration_data()
backend_conf.set('ENABLE_BACKEND_USB', true)
backend_conf.set('ENABLE_BACKEND_LIBUSB', true)
backend_conf.set('ENABLE_BACKEND_CYAPI', false)
backend_conf.set('ENABLE_BACKEND_DUMMY', false)
backend_conf.set('ENABLE_BACKEND_LINUX_DRIVER', false)
subdir('host/libraries/libbladeRF/src/backend')

subdir('ad936x')

private_args = [
  '-DLOGGING_ENABLED',
  '-DHAVE_LIBUSB_GET_VERSION',
  '-DNUAND_MODIFICATIONS',
  '-DLIBBLADERF_SEARCH_PREFIX="' + get_option('prefix').replace('\\', '\\\\') + '"',
]
if system == 'linux'
  private_args += ['-DENABLE_USB_DEV_RESET_ON_OPEN=1']
endif
if is_windows
  private_args += ['-DWIN32', '-D_CRT_SECURE_NO_WARNINGS']
endif

private_inc = [ad936x_inc, include_directories(
  '.',
  'host/libraries/libbladeRF/include',
  'host/libraries/libbladeRF/src',
  'host/common/include',
  'firmware_common',
  'fpga_common/include',
  'thirdparty/analogdevicesinc/no-OS_local/platform_bladerf2',
)]
if system == 'darwin'
  private_inc += include_directories('host/common/include/osx')
elif is_windows
  private_inc += include_directories('host/common/include/windows')
endif

sources = files(
  'host/libraries/libbladeRF/src/backend/backend.c',
  'host/libraries/libbladeRF/src/backend/usb/nios_access.c',
  'host/libraries/libbladeRF/src/backend/usb/nios_legacy_access.c',
  'host/libraries/libbladeRF/src/backend/usb/usb.c',
  'host/libraries/libbladeRF/src/backend/usb/libusb.c',
  'host/libraries/libbladeRF/src/driver/spi_flash.c',
  'host/libraries/libbladeRF/src/driver/fx3_fw.c',
  'host/libraries/libbladeRF/src/driver/fpga_trigger.c',
  'host/libraries/libbladeRF/src/driver/si5338.c',
  'host/libraries/libbladeRF/src/driver/ina219.c',
  'host/libraries/libbladeRF/src/driver/dac161s055.c',
  'host/libraries/libbladeRF/src/driver/smb_clock.c',
  'host/libraries/libbladeRF/src/board/bladerf1/bladerf1.c',
  'host/libraries/libbladeRF/src/board/bladerf1/capabilities.c',
  'host/libraries/libbladeRF/src/board/bladerf1/compatibility.c',
  'host/libraries/libbladeRF/src/board/bladerf1/calibration.c',
  'host/libraries/libbladeRF/src/board/bladerf1/flash.c',
  'host/libraries/libbladeRF/src/board/bladerf1/image.c',
  'host/libraries/libbladeRF/src/board/bladerf2/bladerf2.c',
  'host/libraries/libbladeRF/src/board/bladerf2/capabilities.c',
  'host/libraries/libbladeRF/src/board/bladerf2/common.c',
  'host/libraries/libbladeRF/src/board/bladerf2/compatibility.c',
  'host/libraries/libbladeRF/src/board/bladerf2/rfic_fpga.c',
  'host/libraries/libbladeRF/src/board/bladerf2/rfic_host.c',
  'host/libraries/libbladeRF/src/board/board.c',
  'host/libraries/libbladeRF/src/expansion/xb100.c',
  'host/libraries/libbladeRF/src/expansion/xb200.c',
  'host/libraries/libbladeRF/src/expansion/xb300.c',
  'host/libraries/libbladeRF/src/streaming/async.c',
  'host/libraries/libbladeRF/src/streaming/sync.c',
  'host/libraries/libbladeRF/src/streaming/sync_worker.c',
  'host/libraries/libbladeRF/src/init_fini.c',
  'host/libraries/libbladeRF/src/helpers/timeout.c',
  'host/libraries/libbladeRF/src/helpers/file.c',
  'host/libraries/libbladeRF/src/helpers/version.c',
  'host/libraries/libbladeRF/src/helpers/wallclock.c',
  'host/libraries/libbladeRF/src/helpers/interleave.c',
  'host/libraries/libbladeRF/src/helpers/configfile.c',
  'host/libraries/libbladeRF/src/devinfo.c',
  'host/libraries/libbladeRF/src/device_calibration.c',
  'host/libraries/libbladeRF/src/bladerf.c',
  'host/common/src/sha256.c',
  'host/common/src/conversions.c',
  'host/common/src/log.c',
  'host/common/src/parse.c',
  'host/common/src/range.c',
  'fpga_common/src/lms.c',
  'fpga_common/src/band_select.c',
  'fpga_common/src/ad936x_helpers.c',
  'fpga_common/src/bladerf2_common.c',
  'fpga_common/src/ad936x_params.c',
  'thirdparty/analogdevicesinc/no-OS_local/platform_bladerf2/platform.c',
  'thirdparty/analogdevicesinc/no-OS_local/platform_bladerf2/adc_core.c',
  'thirdparty/analogdevicesinc/no-OS_local/platform_bladerf2/dac_core.c',
)
if is_windows and (cc.get_id() == 'msvc' or cc.get_id() == 'clang-cl')
  sources += files('host/common/src/windows/setenv.c')
  if not have_clock_gettime
    sources += files('host/common/src/windows/clock_gettime.c')
  endif
endif
if system == 'darwin' and not have_clock_gettime
  sources += files('host/common/src/osx/clock_gettime.c')
endif

libbladerf_lib = library(
  'bladeRF',
  sources,
  ad936x_sources,
  c_args: private_args,
  include_directories: private_inc,
  dependencies: deps,
  gnu_symbol_visibility: 'hidden',
  install: true,
)

libbladerf_dep = declare_dependency(
  include_directories: ['host/libraries/libbladeRF/include'],
  link_with: libbladerf_lib,
  dependencies: deps,
)
