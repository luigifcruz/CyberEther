src_lst += files([
  'base.cc',
  'helpers.cc',
])

executable(
  'bench',
  src_lst + ['benchmark.cc'],
  include_directories: inc_lst,
  dependencies: dep_lst,
)
