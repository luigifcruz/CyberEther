if py.found()
    flowgraph_examples = custom_target(
        input : files([
            meson.project_source_root() / 'examples/flowgraphs/spectrum-analyzer.yml',
        ]),
        output : ['examples.cc'],
        command : [py, files('parser.py'), meson.project_build_root(), '@OUTPUT@', '@INPUT@'],
    )
    src_lst += flowgraph_examples
endif
