project(
  'libhackrf',
  'c',
  version: '2026.01.3',
  default_options: [
    'optimization=2',
    'default_library=static',
  ],
)

libusb_dep = dependency('libusb-1.0', required: false)
if not libusb_dep.found()
  libusb_dep = subproject(
    'libusb',
    required: true,
    default_options: [
      'default_library=static',
      'examples=false',
      'tests=false',
    ],
  ).get_variable('libusb_dep')
endif

deps = [libusb_dep]
private_c_args = []
inc_dirs = [
  include_directories('host/libhackrf/src'),
]

if host_machine.system() == 'windows'
  private_c_args += ['-DPTW32_STATIC_LIB=1']
  inc_dirs += [include_directories('meson/windows-pthread-shim')]
endif

srcs = files('host/libhackrf/src/hackrf.c')

if host_machine.system() == 'windows'
  srcs += files('meson/windows-pthread-shim/pthread_shim.c')
endif

libhackrf_lib = library(
  'hackrf',
  srcs,
  c_args: private_c_args,
  include_directories: inc_dirs,
  dependencies: deps,
  install: true,
)

public_inc_dirs = [include_directories('host/libhackrf/src')]
if host_machine.system() == 'windows'
  public_inc_dirs += [include_directories('meson/windows-pthread-shim')]
endif

libhackrf_dep = declare_dependency(
  include_directories: public_inc_dirs,
  link_with: libhackrf_lib,
  dependencies: deps,
)
