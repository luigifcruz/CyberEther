if not py.found()
    error('Python is required to generate the runtime Python bridge header.')
endif

runtime_python_bridge = custom_target(
    'runtime_python_bridge',
    input: files([
        'console.py',
        'tensor.py',
        'runtime.py',
    ]),
    output: 'bridge.hh',
    command: [py, files('meson.py'), '@OUTPUT@', '@INPUT@'],
)

dep_lst += declare_dependency(
    sources: runtime_python_bridge,
)
