if python_bin.found()
    resources = custom_target(
        input : files([
            'compositor_banner.jpg',
        ]),
        output : ['resources.hh'],
        command : [python_bin, files('parser.py'), meson.project_build_root(), '@OUTPUT@', '@INPUT@'],
    )

    dep_lst += declare_dependency(
        sources : resources,
    )
endif