shader_pkg_lst += [{
    'name': 'global',
    'includes': files('../common/noise.glsl'),
    'shaders': [
        files(['text.vert.glsl', 'text.frag.glsl']),
        files(['shapes.vert.glsl', 'shapes.frag.glsl']),
        files(['grid.vert.glsl', 'grid.frag.glsl']),
        files(['mesh_color.vert.glsl', 'mesh_color.frag.glsl']),
        files(['curves.vert.glsl', 'curves.frag.glsl']),
        files(['image.vert.glsl', 'image.frag.glsl']),
        files(['punch.vert.glsl', 'punch.frag.glsl']),
        files(['present.vert.glsl', 'present.frag.glsl']),
        files(['orb.vert.glsl', 'orb.frag.glsl']),
    ],
    'kernels': [
        files(['thicklinestrip.comp']),
        files(['thicklines.comp']),
        files(['shapes.comp']),
        files(['fillarea.comp']),
    ],
}]
