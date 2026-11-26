remote_opt = get_option('remote')

deps = []
all_deps_found = remote_opt.enabled() and not jst_is_browser

if all_deps_found
    ffmpeg_cpu = host_machine.cpu_family()
    ffmpeg_nvidia = (jst_is_linux or jst_is_windows) and ['x86_64', 'aarch64'].contains(ffmpeg_cpu)

    ffmpeg_options = [
        'auto_features=disabled',
        'default_library=static',
        'gpl=disabled',
        'version3=disabled',
        'nonfree=disabled',
        'programs=disabled',
        'ffmpeg=disabled',
        'ffplay=disabled',
        'ffprobe=disabled',
        'tests=disabled',
        'checkasm=disabled',
        'avdevice=disabled',
        'avfilter=enabled',
        'avformat=disabled',
        'swresample=disabled',
        'postproc=disabled',
        'swscale=enabled',
        'network=disabled',
        'devices=disabled',
        'encoders=disabled',
        'decoders=disabled',
        'parsers=disabled',
        'bsfs=disabled',
        'hwaccels=disabled',
        'muxers=disabled',
        'demuxers=disabled',
        'protocols=disabled',
        'filters=disabled',
        'hwmap_filter=enabled',
        'indevs=disabled',
        'outdevs=disabled',
        'appkit=disabled',
        'audiotoolbox=disabled',
        'avfoundation=disabled',
        'bzlib=disabled',
        'coreimage=disabled',
        'iconv=disabled',
        'lzma=disabled',
        'metal=disabled',
        'securetransport=disabled',
        'vulkan=disabled',
        'zlib=disabled',

        'libopenh264=enabled',
        'libopenh264_encoder=enabled',

        'openh264:default_library=static',
        'openh264:tests=disabled',
    ]

    if jst_is_windows
        ffmpeg_options += ['w32threads=enabled']
    else
        ffmpeg_options += ['pthreads=enabled']
    endif

    if ['x86', 'x86_64'].contains(ffmpeg_cpu)
        ffmpeg_options += ['x86asm=enabled']
    endif

    if jst_is_macos
        ffmpeg_options += [
            'videotoolbox=enabled',
            'h264_videotoolbox_encoder=enabled',
            'hevc_videotoolbox_encoder=enabled',
        ]
    endif

    if ffmpeg_nvidia
        ffmpeg_options += [
            'ffnvcodec=enabled',
            'nvenc=enabled',
            'cuda=enabled',
            'h264_nvenc_encoder=enabled',
            'hevc_nvenc_encoder=enabled',
            'av1_nvenc_encoder=enabled',
        ]
    endif

    if jst_is_linux
        ffmpeg_options += [
            'v4l2_m2m=enabled',
            'h264_v4l2m2m_encoder=enabled',
            'hevc_v4l2m2m_encoder=enabled',
        ]
    endif

    if jst_is_windows
        ffmpeg_options += [
            'd3d11va=enabled',
            'amf=enabled',
            'h264_amf_encoder=enabled',
            'hevc_amf_encoder=enabled',
            'av1_amf_encoder=enabled',
            'libvpl=enabled',
            'h264_qsv_encoder=enabled',
            'hevc_qsv_encoder=enabled',
            'av1_qsv_encoder=enabled',
            'vpp_qsv_filter=enabled',
        ]
        subproject('amf-headers', required: true)
        subproject('libvpl', required: true)
    endif

    if jst_is_linux
        ffmpeg_options += [
            'libdrm=enabled',
            'vaapi=enabled',
            'vaapi_drm=enabled',
            'h264_vaapi_encoder=enabled',
            'hevc_vaapi_encoder=enabled',
            'av1_vaapi_encoder=enabled',
            'scale_vaapi_filter=enabled',
        ]
        subproject('libdrm', required: true, default_options: [
            'default_library=static',
            'intel=disabled',
            'radeon=disabled',
            'amdgpu=disabled',
            'nouveau=disabled',
            'vmwgfx=disabled',
            'omap=disabled',
            'exynos=disabled',
            'freedreno=disabled',
            'tegra=disabled',
            'vc4=disabled',
            'etnaviv=disabled',
            'cairo-tests=disabled',
            'man-pages=disabled',
            'valgrind=disabled',
            'install-test-programs=false',
            'udev=false',
            'tests=false',
        ])
        subproject('libva', required: true)
    endif

    if ffmpeg_nvidia
        subproject('nv-codec-headers', required: true)
    endif

    subproject('ffmpeg', required: true, default_options: ffmpeg_options)

    foreach ffmpeg_lib : ['libavcodec', 'libavfilter', 'libavutil', 'libswscale']
        deps += dependency(ffmpeg_lib, required: true, static: true).partial_dependency(
            compile_args: true,
            includes: true,
            links: true,
            sources: true,
        )
    endforeach

    if jst_is_linux
        deps += dependency('libdrm', required: true).partial_dependency(compile_args: true, includes: true)
        deps += dependency('libva', required: true).partial_dependency(compile_args: true, includes: true, sources: true)
    endif

    if jst_is_linux and ffmpeg_nvidia
        deps += dependency('ffnvcodec', required: true).partial_dependency(compile_args: true, includes: true)
        cfg_lst.set('JETSTREAM_LOADER_FFMPEG_CUDA_AVAILABLE', true)
    endif

    deps += declare_dependency(
        link_args: meson.get_compiler('c').get_supported_link_arguments(
            '-Wl,--exclude-libs,' + ':'.join([
                'libavcodec.a',
                'libavfilter.a',
                'libavutil.a',
                'libswscale.a',
                'libopenh264.a',
                'libva.a',
                'libva-drm.a',
                'libdrm.a',
            ]),
        ),
    )

    foreach x_dep : deps
        all_deps_found = all_deps_found and x_dep.found()
    endforeach
endif

if all_deps_found
    cfg_lst.set('JETSTREAM_LOADER_FFMPEG_AVAILABLE', true)
    dep_lst += deps
endif

ldr_lst += {'FFmpeg': all_deps_found}
