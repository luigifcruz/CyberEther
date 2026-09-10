shader_pkg_lst += [{
    'name': 'map',
    'includes': files('line_clip.glsl', 'camera.glsl', 'marker.glsl', 'sun.glsl'),
    'msl_args': ['--msl-decoration-binding'],
    'shaders': [
        files(['geoline.vert.glsl', 'geoline.frag.glsl']),
        files(['airport.vert.glsl', 'airport.frag.glsl']),
        files(['citymarker.vert.glsl', 'citymarker.frag.glsl']),
        files(['citydot.vert.glsl', 'citydot.frag.glsl']),
        files(['fill.vert.glsl', 'fill.frag.glsl']),
        files(['atmosphere.vert.glsl', 'atmosphere.frag.glsl']),
        files(['daylight.vert.glsl', 'daylight.frag.glsl']),
        files(['urbanglow.vert.glsl', 'urbanglow.frag.glsl']),
        files(['citylight.vert.glsl', 'citylight.frag.glsl']),
        files(['stars.vert.glsl', 'stars.frag.glsl']),
    ],
    'kernels': [],
}]
