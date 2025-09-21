src_lst += files([
    'base.cc',
])

if python_bin.found()
    flowchart = custom_target(
        input : files([
            meson.project_source_root() / 'examples/flowgraphs/multi-device.yml',
            meson.project_source_root() / 'examples/flowgraphs/multi-fm.yml',
            meson.project_source_root() / 'examples/flowgraphs/overlap-add-fold.yml',
            meson.project_source_root() / 'examples/flowgraphs/overlap-add.yml',
            meson.project_source_root() / 'examples/flowgraphs/remote-instance.yml',
            meson.project_source_root() / 'examples/flowgraphs/remote-spectrum-analyzer.yml',
            meson.project_source_root() / 'examples/flowgraphs/simple-fm.yml',
            meson.project_source_root() / 'examples/flowgraphs/spectrum-analyzer.yml',
        ]),
        output : ['flowgraph_manifest.hh'],
        command : [python_bin, files('flowgraph_parser.py'), meson.project_build_root(), '@OUTPUT@', '@INPUT@'],
    )

    dep_lst += declare_dependency(
        sources : flowchart,
    )
endif
