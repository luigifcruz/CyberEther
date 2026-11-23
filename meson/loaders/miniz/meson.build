miniz_dep = declare_dependency(
    compile_args: ['-DMINIZ_NO_ZLIB_COMPATIBLE_NAMES'],
    dependencies: subproject(
        'miniz',
        required: true,
        default_options: [
            'default_library=static',
        ]
    ).get_variable('miniz_dep'),
)

dep_lst += [miniz_dep]
cfg_lst.set('JETSTREAM_LOADER_MINIZ_AVAILABLE', true)
ldr_lst += {'miniz': miniz_dep.found()}
