project('libairspyhf', ['c'], version: '1.6.8', default_options: ['c_std=c11', 'optimization=2'])

c = meson.get_compiler('c')

if target_machine.system() == 'emscripten'
  dep_lst = [
    dependency('libusb-browser', required: true),
  ]
else
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
  ]
endif

if target_machine.system().startswith('windows') and (c.get_id() == 'msvc' or c.get_id() == 'clang-cl')
  dep_lst += [
    subproject(
      'pthreads4w',
      required: true,
    ).get_variable('pthreads4w_dep'),
  ]
else
  dep_lst += [dependency('threads')]
endif

dep_lst += [c.find_library('m', required: false)]

libairspyhf_lib = library(
  'libairspyhf',
  'libairspyhf/src/airspyhf.c',
  'libairspyhf/src/iqbalancer.c',
  include_directories: ['libairspyhf/src'],
  dependencies: dep_lst,
  install: true,
)

libairspyhf_dep = declare_dependency(
  include_directories: ['libairspyhf/src'],
  link_with: libairspyhf_lib,
  dependencies: dep_lst,
)
