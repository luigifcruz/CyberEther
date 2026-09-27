project(
  'tree-sitter-bash',
  'c',
  version: '0.25.1',
)

tree_sitter_bash_inc = include_directories('bindings/c')

tree_sitter_bash_lib = static_library(
  'tree-sitter-bash',
  files([
    'src/parser.c',
    'src/scanner.c',
  ]),
)

tree_sitter_bash_dep = declare_dependency(
  include_directories: tree_sitter_bash_inc,
  link_with: tree_sitter_bash_lib,
)
