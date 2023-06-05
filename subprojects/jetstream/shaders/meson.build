fs = import('fs')

glslang = find_program('glslangValidator', required : false)
spirvcross = find_program('spirv-cross', required : false)
python_bin = find_program('python3', 'python', required : false)

shader_pkg_lst = []

subdir('constellation')
subdir('lineplot')
subdir('spectrogram')
subdir('waterfall')

foreach shader_pkg : shader_pkg_lst
    # Translate to SPIR-V (Vulkan).
  
    if glslang.found()
        vert = custom_target(
            input : shader_pkg.get('vert'),
            output : fs.stem(shader_pkg.get('vert')) + '.spv',
            command : [glslang, '--quiet', '-V', '-S', 'vert', '@INPUT@', '-o', '@OUTPUT@'],
            build_by_default : true,
        )

        frag = custom_target(
            input : shader_pkg.get('frag'),
            output : fs.stem(shader_pkg.get('frag')) + '.spv',
            command : [glslang, '--quiet', '-V', '-S', 'frag', '@INPUT@', '-o', '@OUTPUT@'],
            build_by_default : true,
        )
    endif

    # Translate to MSL (Metal).
  
    if glslang.found()
        custom_target(
            input : vert,
            output : fs.stem(shader_pkg.get('vert')) + '.msl',
            command : [spirvcross, '--msl', '@INPUT@', '--output', '@OUTPUT@'],
            build_by_default : true,
        )

        custom_target(
            input : frag,
            output : fs.stem(shader_pkg.get('frag')) + '.msl',
            command : [spirvcross, '--msl', '@INPUT@', '--output', '@OUTPUT@'],
            build_by_default : true,
        )
    endif

    # Package all translations into a single header.

    if python_bin.found()
        basename = fs.stem(fs.stem(shader_pkg.get('vert')))
        custom_target(
            input : [vert, frag],
            output : basename + '_shaders.hh',
            command : [python_bin, files('parser.py'), meson.project_build_root(), basename],
            build_by_default : true,
        )
    endif
  
endforeach
