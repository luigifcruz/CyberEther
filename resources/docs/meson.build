if py.found()
    docs_bundle = custom_target(
        input : files([
            meson.project_source_root() / 'docs/acknowledgments.md',
            meson.project_source_root() / 'docs/architecture.md',
            meson.project_source_root() / 'docs/blocks-and-modules.md',
            meson.project_source_root() / 'docs/blocks.md',
            meson.project_source_root() / 'docs/contributing.md',
            meson.project_source_root() / 'docs/faq.md',
            meson.project_source_root() / 'docs/installation.md',
            meson.project_source_root() / 'docs/installing-plugins.md',
            meson.project_source_root() / 'docs/license.md',
            meson.project_source_root() / 'docs/metadata.md',
            meson.project_source_root() / 'docs/native.md',
            meson.project_source_root() / 'docs/overview.md',
            meson.project_source_root() / 'docs/platforms.md',
            meson.project_source_root() / 'docs/plugins.md',
            meson.project_source_root() / 'docs/python-block.md',
            meson.project_source_root() / 'docs/quick-start.md',
            meson.project_source_root() / 'docs/remote.md',
            meson.project_source_root() / 'docs/tensors.md',
            meson.project_source_root() / 'docs/web.md',
        ]),
        output : ['docs.cc'],
        command : [py, files('parser.py'), '@OUTPUT@', '@INPUT@'],
    )
    src_lst += docs_bundle
endif
