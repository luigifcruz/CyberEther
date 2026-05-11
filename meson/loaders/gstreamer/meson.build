remote_opt = get_option('remote')

deps = []
all_deps_found = false

if remote_opt.enabled()
    gst_full_libraries = [
        'app',
        'rtp',
        'sdp',
        'video',
        'webrtc',
    ]

    gst_full_elements = [
        'app:appsrc',
        'coreelements:capsfilter,queue,tee',
        'openh264:openh264enc',
        'rawparse:rawvideoparse',
        'rtp:rtph264pay,rtpvp8pay,rtpvp9pay',
        'videoconvertscale:videoconvert',
        'videoparsersbad:h264parse',
        'vpx:vp8enc,vp9enc',
    ]

    gst_plugin_suffix = '.a'
    if jst_is_windows
        gst_plugin_suffix = '.lib'
    endif

    gst_full_plugins = [
        'libgstapp' + gst_plugin_suffix,
        'libgstcoreelements' + gst_plugin_suffix,
        'libgstdtls' + gst_plugin_suffix,
        'libgstopenh264' + gst_plugin_suffix,
        'libgstrawparse' + gst_plugin_suffix,
        'libgstrtp' + gst_plugin_suffix,
        'libgstsctp' + gst_plugin_suffix,
        'libgstsrtp' + gst_plugin_suffix,
        'libgstvideoconvertscale' + gst_plugin_suffix,
        'libgstvideoparsersbad' + gst_plugin_suffix,
        'libgstvpx' + gst_plugin_suffix,
        'libgstwebrtc' + gst_plugin_suffix,
    ]

    gst_platform_options = []

    if jst_is_macos
        gst_full_elements += ['applemedia:vtenc_h264_hw']
        gst_full_plugins += ['libgstapplemedia' + gst_plugin_suffix]
        gst_platform_options += ['gst-plugins-bad:applemedia=enabled']
    else
        gst_platform_options += ['gst-plugins-bad:applemedia=disabled']
    endif

    if jst_is_linux
        gst_full_elements += [
            'nvcodec:nvh264enc',
            'v4l2codecs:v4l2h264enc',
        ]
        gst_full_plugins += [
            'libgstnvcodec' + gst_plugin_suffix,
            'libgstv4l2codecs' + gst_plugin_suffix,
        ]
        gst_platform_options += [
            'gst-plugins-bad:nvcodec=enabled',
            'gst-plugins-bad:v4l2codecs=enabled',
        ]
    else
        gst_platform_options += [
            'gst-plugins-bad:nvcodec=disabled',
            'gst-plugins-bad:v4l2codecs=disabled',
        ]
    endif

    gst_disabled_base_plugins = [
        'adder', 'alsa', 'audioconvert', 'audiomixer', 'audiorate',
        'audioresample', 'audiotestsrc', 'cdparanoia', 'compositor',
        'debugutils', 'drm', 'dsd', 'encoding', 'gio', 'gio-typefinder',
        'gl-graphene', 'gl-jpeg', 'gl-png', 'iso-codes',
        'libvisual', 'ogg', 'opus', 'overlaycomposition', 'pango',
        'pbtypes', 'playback', 'subparse', 'tcp', 'theora', 'tremor',
        'typefind', 'videorate', 'videotestsrc', 'volume', 'vorbis',
        'x11', 'xi', 'xshm', 'xvideo',
    ]

    gst_disabled_good_plugins = [
        'aalib', 'adaptivedemux2', 'alpha', 'amrnb', 'amrwbdec',
        'apetag', 'audiofx', 'audioparsers', 'auparse', 'autodetect',
        'avi', 'bz2', 'cairo', 'cutter', 'debugutils', 'deinterlace',
        'directsound', 'dtmf', 'dv', 'dv1394', 'effectv', 'equalizer',
        'flac', 'flv', 'flx', 'gdk-pixbuf', 'goom', 'goom2k1', 'gtk3',
        'icydemux', 'id3demux', 'imagefreeze', 'interleave', 'isomp4',
        'jack', 'jpeg', 'lame', 'law', 'level', 'libcaca', 'matroska',
        'monoscope', 'mpg123', 'multifile', 'multipart', 'oss', 'oss4',
        'osxaudio', 'osxvideo', 'png', 'pulse', 'qt-egl', 'qt-wayland',
        'qt-x11', 'qt5', 'qt6', 'replaygain', 'rpicamsrc', 'rtpmanager',
        'rtsp', 'shapewipe', 'shout2', 'smpte', 'soup', 'speex',
        'spectrum', 'taglib', 'twolame', 'udp', 'v4l2', 'videobox',
        'videocrop', 'videofilter', 'videomixer', 'waveform', 'wavenc',
        'wavpack', 'wavparse', 'ximagesrc', 'xingmux', 'y4m',
    ]

    gst_disabled_bad_plugins = [
        'accurip', 'adpcmdec', 'adpcmenc', 'aes', 'aiff', 'aja', 'amfcodec',
        'analyticsoverlay', 'androidmedia', 'aom', 'asio', 'asfmux',
        'assrender', 'audiofxbad', 'audiobuffersplit', 'audiolatency',
        'audiomixmatrix', 'audiovisualizers', 'autoconvert', 'avtp', 'bayer',
        'bluez', 'bs2b', 'bz2', 'camerabin2', 'chromaprint', 'closedcaption',
        'codecalpha', 'codec2json', 'codectimestamper', 'coloreffects',
        'colormanagement', 'curl', 'curl-ssh2', 'd3d11', 'd3d12',
        'd3dvideosink', 'dash', 'dc1394', 'decklink', 'debugutils',
        'directfb', 'directshow', 'directsound', 'drm', 'dts', 'dvb',
        'dvbsubenc', 'dvbsuboverlay', 'dvdspu', 'dwrite', 'faac', 'faad',
        'faceoverlay', 'fbdev', 'fdkaac', 'festival', 'fieldanalysis',
        'flite', 'fluidsynth', 'freeverb', 'frei0r', 'gaudieffects', 'gdp',
        'geometrictransform', 'gl', 'gme', 'gs', 'gsm', 'gtk3', 'hip', 'hls',
        'id3tag', 'insertbin', 'inter', 'interlace', 'ipcpipeline', 'iqa',
        'isac', 'ivfparse', 'ivtc', 'jpegformat', 'jp2kdecimator', 'kms',
        'ladspa', 'ladspa-rdf', 'lc3', 'lcevcdecoder', 'lcevcencoder',
        'ldac', 'libde265', 'librfb', 'lv2', 'mediafoundation', 'microdns',
        'midi', 'modplug', 'mpeghdec', 'mpeg2enc', 'mpegdemux', 'mpegpsmux',
        'mpegtsdemux', 'mpegtsmux', 'mplex', 'msdk', 'mse', 'musepack', 'mxf',
        'neon', 'netsim', 'nvcomp', 'nvdswrapper', 'onnx', 'onvif', 'openal',
        'openaptx', 'opencv', 'openexr', 'openjpeg', 'openmpt', 'openni2',
        'opensles', 'opus', 'pcapparse', 'pnm', 'proxy', 'qroverlay', 'qsv',
        'rawparse', 'removesilence', 'resindvd', 'rist', 'rsvg', 'rtmp',
        'rtmp2', 'rtp', 'sbc', 'sdp', 'segmentclip', 'shm', 'siren', 'smooth',
        'smoothstreaming', 'sndfile', 'soundtouch', 'spandsp', 'speed', 'srt',
        'subenc', 'svtav1', 'svthevcenc', 'svtjpegxs', 'switchbin',
        'teletext', 'tensordecoders', 'tflite', 'tinyalsa', 'timecode',
        'transcode', 'ttml', 'udev', 'unixfd', 'uvcgadget', 'uvch264', 'va',
        'videofilters', 'videoframe_audiolevel', 'videosignal', 'vmaf', 'vmnc',
        'voaacenc', 'voamrwbenc', 'vulkan', 'wasapi', 'wasapi2', 'wayland',
        'webp', 'webrtcdsp', 'webview2', 'wic', 'wildmidi', 'win32ipc',
        'winks', 'winscreencap', 'wpe', 'wpe2', 'x11', 'x265', 'zbar', 'zxing',
    ]

    gst_disabled_base_options = []
    foreach plugin : gst_disabled_base_plugins
        gst_disabled_base_options += ['gst-plugins-base:' + plugin + '=disabled']
    endforeach

    gst_disabled_good_options = []
    foreach plugin : gst_disabled_good_plugins
        gst_disabled_good_options += ['gst-plugins-good:' + plugin + '=disabled']
    endforeach

    gst_disabled_bad_options = []
    foreach plugin : gst_disabled_bad_plugins
        gst_disabled_bad_options += ['gst-plugins-bad:' + plugin + '=disabled']
    endforeach

    gst_default_options = [
        'auto_features=disabled',
        'default_library=static',
        'prefer_static=true',
        'introspection=disabled',
        'nls=disabled',
        'orc=disabled',
        'tests=disabled',
        'tls=disabled',
        'tools=disabled',
        'examples=disabled',

        'base=enabled',
        'good=enabled',
        'bad=enabled',
        'ugly=disabled',
        'libav=disabled',
        'ges=disabled',
        'devtools=disabled',
        'gst-examples=disabled',
        'rtsp_server=disabled',

        'gpl=disabled',
        'gst-plugins-bad:gpl=disabled',
        'gst-plugins-ugly:gpl=disabled',
        'gstreamer:default_library=static',
        'gstreamer:tests=disabled',
        'gst-plugins-base:default_library=static',
        'gst-plugins-base:auto_features=disabled',
        'gst-plugins-base:orc=disabled',
        'gst-plugins-good:default_library=static',
        'gst-plugins-good:auto_features=disabled',
        'gst-plugins-good:orc=disabled',
        'gst-plugins-bad:default_library=static',
        'gst-plugins-bad:auto_features=disabled',
        'gst-plugins-bad:orc=disabled',
        'glib:default_library=static',
        'glib:tests=false',
        'json-glib:tests=false',
        'libnice:crypto-library=openssl',
        'libnice:default_library=static',
        'libnice:gstreamer=disabled',
        'libnice:gupnp=disabled',
        'libnice:introspection=disabled',
        'libnice:tests=disabled',
        'libsrtp2:default_library=static',
        'openh264:default_library=static',
        'orc:default_library=static',
        'pcre2:test=false',
        'proxy-libintl:default_library=static',
        'vpx:default_library=static',

        'gst-full-target-type=static_library',
        'gst-full-libraries=' + ','.join(gst_full_libraries),
        'gst-full-plugins=' + ';'.join(gst_full_plugins),
        'gst-full-elements=' + ';'.join(gst_full_elements),
        'gst-full-typefind-functions=[]',
        'gst-full-device-providers=[]',
        'gst-full-dynamic-types=[]',

        'gst-plugins-base:app=enabled',
        'gst-plugins-base:gl=enabled',
        'gst-plugins-base:rawparse=enabled',
        'gst-plugins-base:videoconvertscale=enabled',
        'gst-plugins-good:rtp=enabled',
        'gst-plugins-good:vpx=enabled',
        'gst-plugins-bad:dtls=enabled',
        'gst-plugins-bad:openh264=enabled',
        'gst-plugins-bad:sctp=enabled',
        'gst-plugins-bad:srtp=enabled',
        'gst-plugins-bad:videoparsers=enabled',
        'gst-plugins-bad:webrtc=enabled',
    ] + gst_platform_options + gst_disabled_base_options + gst_disabled_good_options + gst_disabled_bad_options

    subproject('pcre2', required: true, default_options: [
        'default_library=static',
        'test=false',
    ])

    subproject('glib', required: true, default_options: [
        'default_library=static',
        'introspection=disabled',
        'nls=disabled',
        'proxy-libintl:default_library=static',
        'tests=false',
    ])

    subproject('libsrtp2', required: true, default_options: [
        'default_library=static',
    ])

    subproject('vpx', required: true, default_options: [
        'default_library=static',
    ])

    gst_full_dep = dependency(
        'gstreamer-full-1.0',
        fallback: 'gstreamer-full',
        default_options: gst_default_options,
        required: true,
        static: true,
    )

    deps = [
        gst_full_dep,
        dependency('gstreamer-app-1.0', required: true, static: true),
        dependency('gstreamer-rtp-1.0', required: true, static: true),
        dependency('gstreamer-sdp-1.0', required: true, static: true),
        dependency('gstreamer-video-1.0', required: true, static: true),
        dependency('gstreamer-webrtc-1.0', required: true, static: true),
    ]

    all_deps_found = true
    foreach x_dep : deps
        all_deps_found = all_deps_found and x_dep.found()
    endforeach
endif

if all_deps_found
    cfg_lst.set('JETSTREAM_LOADER_GSTREAMER_AVAILABLE', true)
    dep_lst += deps
endif

ldr_lst += {'GStreamer': all_deps_found}
