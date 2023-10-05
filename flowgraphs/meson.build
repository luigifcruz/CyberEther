if python_bin.found()
    flowchart = custom_target(
        output : 'flowgraphs.hh',
        command : [python_bin, files('parser.py'), meson.project_build_root()],
    )

    dep_lst += declare_dependency(
        sources : flowchart,
    )
endif