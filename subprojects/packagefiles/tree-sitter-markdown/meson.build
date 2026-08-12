project(
  'tree-sitter-markdown',
  'c',
  version: '0.5.3',
)

tree_sitter_markdown_inc = include_directories('tree-sitter-markdown/bindings/c')
tree_sitter_markdown_src_inc = include_directories('tree-sitter-markdown/src')

tree_sitter_markdown_lib = static_library(
  'tree-sitter-markdown',
  files([
    'tree-sitter-markdown/src/parser.c',
    'tree-sitter-markdown/src/scanner.c',
  ]),
  include_directories: tree_sitter_markdown_src_inc,
)

tree_sitter_markdown_dep = declare_dependency(
  include_directories: tree_sitter_markdown_inc,
  link_with: tree_sitter_markdown_lib,
)
