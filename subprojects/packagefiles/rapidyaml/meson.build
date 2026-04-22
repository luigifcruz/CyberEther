project(
  'rapidyaml',
  'cpp',
  version: '0.11.1',
  default_options: ['cpp_std=c++11'],
)

c4core_inc = include_directories('ext/c4core/src')

c4core_lib = static_library(
  'c4core',
  files(
    'ext/c4core/src/c4/base64.cpp',
    'ext/c4core/src/c4/char_traits.cpp',
    'ext/c4core/src/c4/error.cpp',
    'ext/c4core/src/c4/format.cpp',
    'ext/c4core/src/c4/language.cpp',
    'ext/c4core/src/c4/memory_resource.cpp',
    'ext/c4core/src/c4/memory_util.cpp',
    'ext/c4core/src/c4/utf.cpp',
    'ext/c4core/src/c4/version.cpp',
  ),
  include_directories: c4core_inc,
)

c4core_dep = declare_dependency(
  include_directories: c4core_inc,
  link_with: c4core_lib,
)

ryml_inc = include_directories('src')

ryml_lib = static_library(
  'ryml',
  files(
    'src/c4/yml/common.cpp',
    'src/c4/yml/node.cpp',
    'src/c4/yml/node_type.cpp',
    'src/c4/yml/parse.cpp',
    'src/c4/yml/preprocess.cpp',
    'src/c4/yml/reference_resolver.cpp',
    'src/c4/yml/tag.cpp',
    'src/c4/yml/tree.cpp',
    'src/c4/yml/version.cpp',
  ),
  include_directories: [ryml_inc, c4core_inc],
  dependencies: c4core_dep,
)

ryml_dep = declare_dependency(
  include_directories: [ryml_inc, c4core_inc],
  link_with: ryml_lib,
  dependencies: c4core_dep,
)
