python_bin = find_program('python3', 'python', required : false)

if python_bin.found()
    custom_target(
        output : 'flowgraphs.hh',
        command : [python_bin, files('parser.py'), meson.project_build_root()],
    )
endif