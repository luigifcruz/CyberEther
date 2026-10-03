deps = [
    subproject('tree-sitter', required: true, default_options: [
        'default_library=static',
    ]).get_variable('tree_sitter_dep'),
    subproject('tree-sitter-python', required: true).get_variable('tree_sitter_python_dep'),
    subproject('tree-sitter-markdown', required: true).get_variable('tree_sitter_markdown_dep'),
    subproject('tree-sitter-yaml', required: true).get_variable('tree_sitter_yaml_dep'),
    subproject('tree-sitter-bash', required: true).get_variable('tree_sitter_bash_dep'),
    subproject('tree-sitter-cpp', required: true).get_variable('tree_sitter_cpp_dep'),
    subproject('tree-sitter-json', required: true).get_variable('tree_sitter_json_dep'),
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and x_dep.found()
endforeach

if all_deps_found
    cfg_lst.set('JETSTREAM_LOADER_TREE_SITTER_AVAILABLE', true)
    cfg_lst.set('JETSTREAM_LOADER_TREE_SITTER_PYTHON_AVAILABLE', true)
    cfg_lst.set('JETSTREAM_LOADER_TREE_SITTER_MARKDOWN_AVAILABLE', true)
    cfg_lst.set('JETSTREAM_LOADER_TREE_SITTER_YAML_AVAILABLE', true)
    cfg_lst.set('JETSTREAM_LOADER_TREE_SITTER_BASH_AVAILABLE', true)
    cfg_lst.set('JETSTREAM_LOADER_TREE_SITTER_CPP_AVAILABLE', true)
    cfg_lst.set('JETSTREAM_LOADER_TREE_SITTER_JSON_AVAILABLE', true)
    dep_lst += deps
endif

ldr_lst += {'Tree-sitter': all_deps_found}
ldr_lst += {'Tree-sitter Python': all_deps_found}
ldr_lst += {'Tree-sitter Markdown': all_deps_found}
ldr_lst += {'Tree-sitter YAML': all_deps_found}
ldr_lst += {'Tree-sitter Bash': all_deps_found}
ldr_lst += {'Tree-sitter C++': all_deps_found}
ldr_lst += {'Tree-sitter JSON': all_deps_found}
