src_lst += files([
    'axis.cc',
    'font.cc',
    'text.cc',
    'shapes.cc',
    'mesh.cc',
    'image.cc',
    'curves.cc',
    'voice_wave.cc',
    'map_context.cc',
    'map_layer.cc',
])

if cfg_lst.get('JETSTREAM_LOADER_ZLIB_AVAILABLE', false)
    src_lst += files([
        'geomap.cc',
        'geomap_base.cc',
    ])
endif
