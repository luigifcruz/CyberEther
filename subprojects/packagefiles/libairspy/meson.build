project('libairspy', ['c', 'cpp'], version: '1.0.10', default_options: ['c_std=c11', 'cpp_std=c++11', 'optimization=2'])

private_cpp_args = []
interface_cpp_args = []
private_cpp_args += []
interface_cpp_args += []
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

libairspy_lib = library(
  'libairspy',
  'libairspy/src/airspy.c',
  'libairspy/src/iqconverter_float.c',
  'libairspy/src/iqconverter_int16.c',
  cpp_args: private_cpp_args,
  include_directories: ['libairspy/src'],
  dependencies: dep_lst,
  install: true,
)

libairspy_dep = declare_dependency(
  include_directories: ['libairspy/src'],
  compile_args: interface_cpp_args,
  link_with: libairspy_lib,
  dependencies: dep_lst,
)

install_subdir('include/libairspy', install_dir: get_option('includedir'))
