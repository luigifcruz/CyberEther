if not py.found()
    error('Python is required to generate the runtime Python bootstrap header.')
endif

runtime_python_bootstrap_activation = custom_target(
    'runtime_python_bootstrap_activation',
    input: 'activation.py',
    output: 'activation.hh',
    command: [
        py,
        runtime_python_embed_script,
        '@OUTPUT@',
        'kPythonBootstrapActivation',
        'bridge/bootstrap',
        '@INPUT@',
    ],
)

dep_lst += declare_dependency(
    sources: runtime_python_bootstrap_activation,
)

runtime_python_bootstrap_source = custom_target(
    'runtime_python_bootstrap_source',
    input: 'source.py',
    output: 'source.hh',
    command: [
        py,
        runtime_python_embed_script,
        '@OUTPUT@',
        'kPythonBootstrapSource',
        'bridge/bootstrap',
        '@INPUT@',
    ],
)

dep_lst += declare_dependency(
    sources: runtime_python_bootstrap_source,
)

src_lst += files('base.cc')
