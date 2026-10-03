project(
  'tree-sitter-json',
  'c',
  version: '0.24.8',
)

tree_sitter_json_inc = include_directories('bindings/c')

tree_sitter_json_lib = static_library(
  'tree-sitter-json',
  files([
    'src/parser.c',
  ]),
)

tree_sitter_json_dep = declare_dependency(
  include_directories: tree_sitter_json_inc,
  link_with: tree_sitter_json_lib,
)
