if python_bin.found()
    flowchart = custom_target(
        input : files([
            'multi-device.yml',
            'multi-fm.yml',
            'simple-fm.yml',
            'remote-spectrum-analyzer.yml',
            'spectrum-analyzer.yml',
            'speech-recognition-fm.yml',
            'remote-instance.yml',
        ]),
        output : ['data.hh'],
        command : [python_bin, files('parser.py'), meson.project_build_root()],
    )

    dep_lst += declare_dependency(
        sources : flowchart,
    )
endif