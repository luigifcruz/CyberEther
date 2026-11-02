project('birriasdr', ['c'], version: '0.1.0', default_options: ['c_std=c11', 'optimization=2'])

c_args = ['-D_POSIX_C_SOURCE=200809L']
if host_machine.system() == 'darwin'
  c_args += ['-D_DARWIN_C_SOURCE']
endif

dep_lst = [
  subproject(
    'libusb',
    required: true,
    default_options: [
      'default_library=static',
      'examples=false',
      'tests=false',
    ],
  ).get_variable('libusb_dep'),
  dependency('threads'),
]

birriasdr_lib = library(
  'birriasdr',
  'src/device.c',
  'src/io.c',
  'src/usb.c',
  'src/usb_stream.c',
  'src/stream.c',
  'src/chips/mt7921_boot.c',
  'src/chips/mt7921_rf.c',
  'src/chips/mt7921_tune.c',
  'src/chips/mt7921_gain.c',
  'src/chips/mt7921_capture.c',
  'src/chips/mt7921_filter.c',
  'src/chips/mt7921_frame.c',
  'src/chips/mt7925.c',
  c_args: c_args,
  gnu_symbol_visibility: 'hidden',
  include_directories: ['include', 'src'],
  dependencies: dep_lst,
  install: true,
)

birriasdr_dep = declare_dependency(
  include_directories: ['include'],
  link_with: birriasdr_lib,
  dependencies: dep_lst,
)
