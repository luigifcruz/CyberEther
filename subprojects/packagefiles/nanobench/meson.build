project(
  'nanobench',
  'cpp',
  version: '4.3.11',
  default_options: ['cpp_std=c++11'],
)

nanobench_inc = include_directories('src/include')

nanobench_lib = static_library(
  'nanobench',
  files('src/test/app/nanobench.cpp'),
  include_directories: nanobench_inc,
)

nanobench_dep = declare_dependency(
  include_directories: nanobench_inc,
  link_with: nanobench_lib,
)
