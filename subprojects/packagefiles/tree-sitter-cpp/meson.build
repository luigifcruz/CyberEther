project(
  'tree-sitter-cpp',
  'c',
  version: '0.23.4',
)

tree_sitter_cpp_inc = include_directories('bindings/c')

tree_sitter_cpp_lib = static_library(
  'tree-sitter-cpp',
  files([
    'src/parser.c',
    'src/scanner.c',
  ]),
)

tree_sitter_cpp_dep = declare_dependency(
  include_directories: tree_sitter_cpp_inc,
  link_with: tree_sitter_cpp_lib,
)
