deps = [
    subproject('tree-sitter', required: true, default_options: [
        'default_library=static',
    ]).get_variable('tree_sitter_dep'),
    subproject('tree-sitter-python', required: true).get_variable('tree_sitter_python_dep'),
    subproject('tree-sitter-markdown', required: true).get_variable('tree_sitter_markdown_dep'),
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and x_dep.found()
endforeach

if all_deps_found
    cfg_lst.set('JETSTREAM_LOADER_TREE_SITTER_AVAILABLE', true)
    cfg_lst.set('JETSTREAM_LOADER_TREE_SITTER_PYTHON_AVAILABLE', true)
    cfg_lst.set('JETSTREAM_LOADER_TREE_SITTER_MARKDOWN_AVAILABLE', true)
    dep_lst += deps
endif

ldr_lst += {'Tree-sitter': all_deps_found}
ldr_lst += {'Tree-sitter Python': all_deps_found}
ldr_lst += {'Tree-sitter Markdown': all_deps_found}
