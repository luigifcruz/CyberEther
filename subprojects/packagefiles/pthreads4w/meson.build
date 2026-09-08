project('pthreads4w', 'c', version: '3.0.0', default_options: ['optimization=2'])

if not target_machine.system().startswith('windows')
  error('pthreads4w is only supported on Windows')
endif

private_c_args = [
  '-DHAVE_CONFIG_H',
  '-D__PTW32_STATIC_LIB',
  '-D__PTW32_BUILD_INLINED',
  '-D__PTW32_CLEANUP_C',
]
interface_c_args = [
  '-D__PTW32_STATIC_LIB',
  '-D__PTW32_CLEANUP_C',
]

pthreads4w_lib = static_library(
  'pthreadVC3',
  'pthread.c',
  c_args: private_c_args,
  include_directories: ['.'],
  install: true,
)

pthreads4w_dep = declare_dependency(
  compile_args: interface_c_args,
  include_directories: ['.'],
  link_with: pthreads4w_lib,
)

install_headers('_ptw32.h', 'pthread.h', 'sched.h', 'semaphore.h')
