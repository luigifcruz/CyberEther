project(
  'libhackrf',
  'c',
  version: '2026.01.3',
  default_options: [
    'optimization=2',
    'default_library=static',
  ],
)

libusb_dep = subproject(
  'libusb',
  required: true,
  default_options: [
    'default_library=static',
    'examples=false',
    'tests=false',
  ],
).get_variable('libusb_dep')

deps = [libusb_dep]
c = meson.get_compiler('c')

if target_machine.system().startswith('windows') and (c.get_id() == 'msvc' or c.get_id() == 'clang-cl')
  deps += [
    subproject(
      'pthreads4w',
      required: true,
    ).get_variable('pthreads4w_dep'),
  ]
else
  deps += [dependency('threads')]
endif

srcs = files('host/libhackrf/src/hackrf.c')

libhackrf_lib = library(
  'hackrf',
  srcs,
  include_directories: ['host/libhackrf/src'],
  dependencies: deps,
  install: true,
)

libhackrf_dep = declare_dependency(
  include_directories: ['host/libhackrf/src'],
  link_with: libhackrf_lib,
  dependencies: deps,
)
