project(
  'tree-sitter-python',
  'c',
  version: '0.25.0',
)

tree_sitter_python_inc = include_directories('bindings/c')

tree_sitter_python_lib = static_library(
  'tree-sitter-python',
  files([
    'src/parser.c',
    'src/scanner.c',
  ]),
)

tree_sitter_python_dep = declare_dependency(
  include_directories: tree_sitter_python_inc,
  link_with: tree_sitter_python_lib,
)
