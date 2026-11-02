project('libiio', 'c', version: '0.26', default_options: ['c_std=gnu11,c11', 'optimization=2'])

cc = meson.get_compiler('c')
system = target_machine.system()
is_windows = system.startswith('windows')

usb_dep = subproject(
  'libusb',
  required: true,
  default_options: ['default_library=static', 'examples=false', 'tests=false'],
).get_variable('libusb_dep')

xml_dep = subproject(
  'libxml2',
  required: true,
  default_options: [
    'default_library=static',
    'minimum=true',
    'valid=enabled',
    'threads=enabled',
  ],
).get_variable('xml_dep')

deps = [usb_dep, xml_dep, dependency('threads')]

c_args = [
  '-D_POSIX_C_SOURCE=200809L',
  '-D__XSI_VISIBLE=500',
  '-DLIBIIO_EXPORTS=1',
  '-DLIBIIO_STATIC=1',
]
if system == 'darwin'
  c_args += ['-D_DARWIN_C_SOURCE']
endif
if is_windows
  c_args += ['-D_CRT_SECURE_NO_WARNINGS']
endif

check_args = ['-D_POSIX_C_SOURCE=200809L']
if system == 'darwin'
  check_args += ['-D_DARWIN_C_SOURCE']
endif

conf = configuration_data()
conf.set('LIBIIO_VERSION_MAJOR', 0)
conf.set('LIBIIO_VERSION_MINOR', 26)
conf.set('LIBIIO_VERSION_GIT', 'v0.26')
conf.set('LOG_LEVEL', 'Warning')
conf.set('LIBIIO_SCAN_BACKENDS', 'usb')
conf.set10('WITH_LOCAL_BACKEND', false)
conf.set10('WITH_XML_BACKEND', true)
conf.set10('WITH_NETWORK_BACKEND', false)
conf.set10('WITH_USB_BACKEND', true)
conf.set10('WITH_SERIAL_BACKEND', false)
conf.set10('WITH_NETWORK_EVENTFD', false)
conf.set10('WITH_IIOD_USBD', false)
conf.set10('WITH_IIOD_SERIAL', false)
conf.set10('WITH_LOCAL_CONFIG', false)
conf.set10('WITH_LOCAL_MMAP_API', false)
conf.set10('WITH_HWMON', false)
conf.set10('WITH_AIO', false)
conf.set10('HAVE_DNS_SD', false)
conf.set10('HAVE_AVAHI', false)
conf.set10('WITH_ZSTD', false)
conf.set('HAS_STRDUP', cc.has_function('strdup', prefix: '#include <string.h>', args: check_args))
conf.set('HAS_STRNDUP', cc.has_function('strndup', prefix: '#include <string.h>', args: check_args))
conf.set('HAS_STRERROR_R', cc.has_function('strerror_r', prefix: '#include <string.h>', args: check_args))
conf.set('HAS_STRTOK_R', cc.has_function('strtok_r', prefix: '#include <string.h>', args: check_args))
conf.set('HAS_NEWLOCALE', cc.has_function('newlocale', prefix: '#include <locale.h>', args: check_args))
conf.set('HAS_LIBUSB_GETVERSION', true)
configure_file(
  input: 'iio-config.h.cmakein',
  output: 'iio-config.h',
  configuration: conf,
  format: 'cmake@',
)

sources = files(
  'backend.c',
  'buffer.c',
  'channel.c',
  'context.c',
  'device.c',
  'iiod-client.c',
  'lock.c',
  'scan.c',
  'sort.c',
  'usb.c',
  'utilities.c',
  'xml.c',
)

libiio_lib = static_library(
  'iio',
  sources,
  c_args: c_args,
  dependencies: deps,
  gnu_symbol_visibility: 'hidden',
)

libiio_dep = declare_dependency(
  include_directories: include_directories('.'),
  compile_args: ['-DLIBIIO_STATIC=1'],
  dependencies: deps,
  link_with: libiio_lib,
)
