fs = import('fs')

glslang = find_program('glslangValidator', required : false)
spirvcross = find_program('spirv-cross', required : false)
naga = find_program('naga', required : false)
python_bin = find_program('python3', 'python', required : false)

shader_pkg_lst = []

subdir('constellation')
subdir('lineplot')
subdir('spectrogram')
subdir('waterfall')

foreach shader_pkg : shader_pkg_lst
    name = shader_pkg.get('name')
    shaders = shader_pkg.get('shaders')
    input_lst = []

    foreach shader : shaders

        # Translate to SPIR-V (Vulkan).
  
        if glslang.found()
            vert = custom_target(
                input : shader[0],
                output : name + '_' + fs.stem(shader[0]) + '.spv',
                command : [glslang, '--quiet', '-V', '-S', 'vert', '@INPUT@', '-o', '@OUTPUT@'],
                build_by_default : true,
            )

            frag = custom_target(
                input : shader[1],
                output : name + '_' + fs.stem(shader[1]) + '.spv',
                command : [glslang, '--quiet', '-V', '-S', 'frag', '@INPUT@', '-o', '@OUTPUT@'],
                build_by_default : true,
            )

            input_lst += [vert, frag]
        endif

        # Translate to MSL (Metal).
  
        if spirvcross.found()
            custom_target(
                input : vert,
                output : name + '_' + fs.stem(shader[0]) + '.msl',
                command : [spirvcross, '--msl', '@INPUT@', '--output', '@OUTPUT@', '--msl-version', '30000'],
                build_by_default : true,
            )

            custom_target(
                input : frag,
                output : name + '_' + fs.stem(shader[1]) + '.msl',
                command : [spirvcross, '--msl', '@INPUT@', '--output', '@OUTPUT@', '--msl-version', '30000'],
                build_by_default : true,
            )
        endif

        # Translate to WGSL (WebGPU).

        if naga.found()
            custom_target(
                input : vert,
                output : name + '_' + fs.stem(shader[0]) + '.wgsl',
                command : [naga, '--keep-coordinate-space', '@INPUT@', '@OUTPUT@'],
                build_by_default : true,
            )

            custom_target(
                input : frag,
                output : name + '_' + fs.stem(shader[1]) + '.wgsl',
                command : [naga, '--keep-coordinate-space', '@INPUT@', '@OUTPUT@'],
                build_by_default : true,
            )
        endif

    endforeach

    # Package all translations into a single header.

    if python_bin.found()
        shader_obj = custom_target(
            input : input_lst,
            output : name + '_shaders.hh',
            command : [python_bin, files('parser.py'), meson.project_build_root(), name],
        )

        dep_lst += declare_dependency(
            sources: shader_obj,
        )
    endif
  
endforeach
