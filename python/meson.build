py = import('python').find_installation('python3', required: false)

py_build_path = meson.current_build_dir()
py_install_path = join_paths(py.get_install_dir(), 'superluminal')

if py.found() and cfg_lst.get('JETSTREAM_SUPERLUMINAL_AVAILABLE', false)
    cfg_lst.set('JETSTREAM_SUPERLUMINAL_PYTHON_AVAILABLE', true)

    subdir('superluminal')

    install_subdir(
        'superluminal',
        install_dir: py.get_install_dir(),
        exclude_files: [
            'superluminal.cc',
            'meson.build',
        ]
    )
endif