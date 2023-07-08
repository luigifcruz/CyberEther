project('libusb', ['c', 'cpp'], version: '1.0.26', meson_version: '>=0.49.0')

add_project_arguments('-D_GNU_SOURCE', language: 'c')

cc = meson.get_compiler('c')
pconf = import('pkgconfig')

thread_dep = dependency('threads')

cdata = configuration_data()
cdata.set('HAVE_SYSLOG', cc.has_function('syslog'))
cdata.set('HAVE_CLOCK_GETTIME', true)
cdata.set('HAVE_EVENTFD', false)
cdata.set('HAVE_TIMERFD', false)
cdata.set('HAVE_PIPE2', false)
cdata.set('HAVE_PTHREAD_CONDATTR_SETCLOCK', cc.has_function('pthread_condattr_setclock', dependencies: thread_dep))
cdata.set('HAVE_PTHREAD_SETNAME_NP', cc.has_function('pthread_setname_np', dependencies: thread_dep))
cdata.set('HAVE_PTHREAD_THREADID_NP', cc.has_function('pthread_threadid_np', dependencies: thread_dep))
cdata.set('HAVE_ASM_TYPES_H', cc.has_header('asm/types.h'))
cdata.set('HAVE_NFDS_T', cc.has_header('nfds.h'))
cdata.set('HAVE_STRING_H', cc.has_header('string.h'))
cdata.set('HAVE_SYS_TIME_H', cc.has_header('sys/time.h'))
cdata.set('HAVE_STRUCT_TIMESPEC', cc.has_header_symbol('time.h', 'struct timespec'))
cdata.set('PRINTF_FORMAT(a, b)', '__attribute__ ((__format__ (__printf__, a, b)))')
cdata.set('ENABLE_LOGGING', true)
cdata.set('ENABLE_DEBUG_LOGGING', true)
cdata.set('USE_SYSTEM_LOGGING_FACILITY', false)

incdirs = include_directories(['.', 'libusb', 'libusb/os'])
install_headers('libusb/libusb.h', subdir: 'libusb-1.0')

sources = files(
  'libusb/core.c',
  'libusb/descriptor.c',
  'libusb/hotplug.c',
  'libusb/io.c',
  'libusb/strerror.c',
  'libusb/sync.c',
 )

cdata.set('PLATFORM_POSIX', true)
cdata.set('DEFAULT_VISIBILITY', '__attribute__ ((visibility ("default")))')
sources += files('libusb/os/events_posix.c', 'libusb/os/threads_posix.c')
sources += files('libusb/os/emscripten_webusb.cpp')

cfile = configure_file(
  output: 'config.h',
  configuration: cdata,
)

libusb = library('libusb-1.0',
  sources,
  dependencies: [ thread_dep ],
  include_directories: incdirs,
  version: '0.3.0',
  soversion: '0',
  name_prefix: '',
  gnu_symbol_visibility: 'hidden',
  vs_module_defs: 'libusb/libusb-1.0.def',
  install: true,
)

pconf.generate(
  libusb,
  name: 'libusb-1.0',
  description: 'C API for USB device access from Linux, Mac OS X, Windows, OpenBSD/NetBSD and Solaris userspace',
  subdirs: 'libusb-1.0',
)

libusb_dep = declare_dependency(
  include_directories : incdirs,
  link_with: libusb,
)