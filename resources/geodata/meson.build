if py.found()
    src_lst += custom_target(
        'geodata',
        input : files('manifest.json'),
        output : ['geodata.cc'],
        command : [py, files('fetch.py'), '@INPUT@', '@OUTPUT@',
                   '--cache-dir', meson.project_source_root() / '.cache/geodata'],
    )
endif
