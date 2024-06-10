fs = import('fs')

shader_pkg_lst = []

subdir('global')
subdir('constellation')
subdir('lineplot')
subdir('spectrogram')
subdir('waterfall')
subdir('remote')

foreach shader_pkg: shader_pkg_lst
    name = shader_pkg.get('name')
    shaders = shader_pkg.get('shaders')
    kernels = shader_pkg.get('kernels')
    tracking = []
    shader_lst = []
    kernel_lst = []

    foreach shader: shaders

        #
        # Translate to SPIR-V (Vulkan).
        #

        build_spirv = glslang_bin.found() and python_bin.found()

        if build_spirv
            spirv_vert = custom_target(
                input: shader[0],
                output: name + '_' + fs.stem(shader[0]) + '.spv',
                command: [glslang_bin, '--quiet', '-V', '-S', 'vert', '@INPUT@', '-o', '@OUTPUT@'],
                build_by_default: true,
            )

            spirv_frag = custom_target(
                input: shader[1],
                output: name + '_' + fs.stem(shader[1]) + '.spv',
                command: [glslang_bin, '--quiet', '-V', '-S', 'frag', '@INPUT@', '-o', '@OUTPUT@'],
                build_by_default: true,
            )

            tracking += [spirv_vert, spirv_frag]
            shader_lst += [spirv_vert, spirv_frag]
            cfg_lst.set('JETSTREAM_SHADERS_VULKAN_AVAILABLE', true)
        endif

        #
        # Translate to MSL (Metal).
        #

        build_msl = spirvcross_bin.found() and build_spirv
  
        if build_msl
            msl_vert = custom_target(
                input: spirv_vert,
                output: name + '_' + fs.stem(shader[0]) + '.msl',
                command: [spirvcross_bin, '--msl', '@INPUT@', '--output', '@OUTPUT@', '--msl-version', '30000'],
                build_by_default: true,
            )

            msl_frag = custom_target(
                input: spirv_frag,
                output: name + '_' + fs.stem(shader[1]) + '.msl',
                command: [spirvcross_bin, '--msl', '@INPUT@', '--output', '@OUTPUT@', '--msl-version', '30000'],
                build_by_default: true,
            )

            shader_lst += [msl_vert, msl_frag]
            cfg_lst.set('JETSTREAM_SHADERS_METAL_AVAILABLE', true)
        endif

        #
        # Translate to WGSL (WebGPU).
        #

        build_wgsl = naga_bin.found() and build_spirv

        if build_wgsl
            wgsl_vert = custom_target(
                input: spirv_vert,
                output: name + '_' + fs.stem(shader[0]) + '.wgsl',
                command: [naga_bin, '--keep-coordinate-space', '@INPUT@', '@OUTPUT@'],
                build_by_default: true,
            )

            wgsl_frag = custom_target(
                input: spirv_frag,
                output: name + '_' + fs.stem(shader[1]) + '.wgsl',
                command: [naga_bin, '--keep-coordinate-space', '@INPUT@', '@OUTPUT@'],
                build_by_default: true,
            )

            shader_lst += [wgsl_vert, wgsl_frag]
            cfg_lst.set('JETSTREAM_SHADERS_WEBGPU_AVAILABLE', true)
        endif

    endforeach

    foreach kernel: kernels

        #
        # Translate to SPIR-V (Vulkan).
        #

        build_spirv = glslang_bin.found() and python_bin.found()

        if build_spirv
            spirv = custom_target(
                input: kernel[0],
                output: name + '_' + fs.stem(kernel[0]) + '.comp.spv',
                command: [glslang_bin, '--quiet', '-V', '-S', 'comp', '@INPUT@', '-o', '@OUTPUT@'],
                build_by_default: true,
            )

            tracking += [spirv]
            kernel_lst += [spirv]
            cfg_lst.set('JETSTREAM_SHADERS_VULKAN_AVAILABLE', true)
        endif

        #
        # Translate to MSL (Metal).
        #

        build_msl = spirvcross_bin.found() and build_spirv
  
        if build_msl
            msl = custom_target(
                input: spirv,
                output: name + '_' + fs.stem(kernel[0]) + '.comp.msl',
                command: [spirvcross_bin, '--msl', '--msl-decoration-binding', '@INPUT@', '--output', '@OUTPUT@', '--msl-version', '30000'],
                build_by_default: true,
            )

            kernel_lst += [msl]
            cfg_lst.set('JETSTREAM_SHADERS_METAL_AVAILABLE', true)
        endif

        #
        # Translate to WGSL (WebGPU).
        #

        build_wgsl = naga_bin.found() and build_spirv

        if build_wgsl
            wgsl = custom_target(
                input: spirv,
                output: name + '_' + fs.stem(kernel[0]) + '.comp.wgsl',
                command: [naga_bin, '--keep-coordinate-space', '@INPUT@', '@OUTPUT@'],
                build_by_default: true,
            )

            kernel_lst += [wgsl]
            cfg_lst.set('JETSTREAM_SHADERS_WEBGPU_AVAILABLE', true)
        endif

    endforeach

    #
    # Package all translations into a single header.
    #

    if build_spirv
        shader_obj = custom_target(
            output: name + '_shaders.hh',
            command: [python_bin, files('parser.py'), meson.project_build_root(), name],
            depends: shader_lst + kernel_lst,
        )

        dep_lst += declare_dependency(
            sources: shader_obj,
        )
    endif

endforeach
