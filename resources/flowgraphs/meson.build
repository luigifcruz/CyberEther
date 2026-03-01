if python_bin.found()
    flowchart = custom_target(
        input : files([
            meson.project_source_root() / 'examples/flowgraphs/flight-tracker.yml',
            meson.project_source_root() / 'examples/flowgraphs/multi-fm.yml',
            meson.project_source_root() / 'examples/flowgraphs/overlap-add-fold.yml',
            meson.project_source_root() / 'examples/flowgraphs/overlap-add.yml',
            meson.project_source_root() / 'examples/flowgraphs/simple-fm.yml',
            meson.project_source_root() / 'examples/flowgraphs/spectrum-analyzer.yml',
            meson.project_source_root() / 'examples/flowgraphs/signal-generator.yml',
        ]),
        output : ['base.hh'],
        command : [python_bin, files('parser.py'), meson.project_build_root(), '@OUTPUT@', '@INPUT@'],
    )
    dep_lst += declare_dependency(
        sources : flowchart,
    )
endif
