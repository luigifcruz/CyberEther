if python_bin.found()
    flowchart = custom_target(
        input : files([
            'multi-device.yml',
            'multi-fm.yml',
            'overlap-add-fold.yml',
            'overlap-add.yml',
            'remote-instance.yml',
            'remote-spectrum-analyzer.yml',
            'simple-fm.yml',
            'spectrum-analyzer.yml',
            ## 'speech-recognition-fm.yml',
        ]),
        output : ['manifest.hh'],
        command : [python_bin, files('parser.py'), meson.project_build_root(), '@OUTPUT@', '@INPUT@'],
    )

    dep_lst += declare_dependency(
        sources : flowchart,
    )
endif