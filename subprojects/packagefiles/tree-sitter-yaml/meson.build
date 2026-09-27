project(
  'tree-sitter-yaml',
  'c',
  version: '0.7.2',
)

tree_sitter_yaml_inc = include_directories('bindings/c')

tree_sitter_yaml_lib = static_library(
  'tree-sitter-yaml',
  files([
    'src/parser.c',
    'src/scanner.c',
  ]),
)

tree_sitter_yaml_dep = declare_dependency(
  include_directories: tree_sitter_yaml_inc,
  link_with: tree_sitter_yaml_lib,
)
