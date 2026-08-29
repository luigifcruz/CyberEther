runtime_python_dependency_preflight = custom_target(
    'runtime_python_dependency_preflight',
    input: 'preflight.py',
    output: 'preflight.hh',
    command: [
        py,
        runtime_python_embed_script,
        '@OUTPUT@',
        'kPythonDependencyPreflight',
        'dependencies',
        '@INPUT@',
    ],
)

dep_lst += declare_dependency(
    sources: runtime_python_dependency_preflight,
)

src_lst += files([
    'metadata.cc',
    'policy.cc',
    'preflight.cc',
])
