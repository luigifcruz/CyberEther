deps = [
    dependency('tree-sitter', required: true),
    subproject('tree-sitter-python', required: true).get_variable('tree_sitter_python_dep'),
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and x_dep.found()
endforeach

if all_deps_found
    cfg_lst.set('JETSTREAM_LOADER_TREE_SITTER_AVAILABLE', true)
    cfg_lst.set('JETSTREAM_LOADER_TREE_SITTER_PYTHON_AVAILABLE', true)
    dep_lst += deps
endif

ldr_lst += {'Tree-sitter': all_deps_found}
ldr_lst += {'Tree-sitter Python': all_deps_found}
