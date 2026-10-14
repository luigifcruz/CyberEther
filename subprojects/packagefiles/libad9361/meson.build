project('libad9361', 'c', version: '0.3', default_options: ['c_std=gnu11,c11', 'optimization=2'])

iio_dep = subproject(
  'libiio',
  required: true,
  default_options: ['default_library=static'],
).get_variable('libiio_dep')

exports = ['-DLIBAD9361_EXPORTS=1']

libad9361_lib = static_library(
  'ad9361',
  'ad9361_baseband_auto_rate.c',
  c_args: exports,
  dependencies: iio_dep,
  gnu_symbol_visibility: 'hidden',
)

libad9361_dep = declare_dependency(
  include_directories: include_directories('.'),
  compile_args: exports,
  dependencies: iio_dep,
  link_with: libad9361_lib,
)
