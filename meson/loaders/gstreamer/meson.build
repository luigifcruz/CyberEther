remote_opt = get_option('remote')
remote_cargo_c = find_program('cargo-cbuild', required: false)

if remote_opt.enabled() and not remote_cargo_c.found()
    error('Remote streaming requires cargo-c to build gst-plugins-rs. Install it with: cargo install cargo-c')
endif

if remote_opt.allowed() and remote_cargo_c.found()
    cpp = meson.get_compiler('cpp')

    remote_gstreamer_options = [
        'default_library=static',
        'auto_features=disabled',
        'gpl=disabled',
        'base=enabled',
        'good=enabled',
        'bad=enabled',
        'ugly=disabled',
        'libav=disabled',
        'rs=enabled',
        'gst-full=enabled',
        'gst-full-target-type=static_library',
        'gst-full-libraries=gstreamer-app-1.0,gstreamer-video-1.0,gstreamer-webrtc-1.0',
        'devtools=disabled',
        'ges=disabled',
        'rtsp_server=disabled',
        'gst-examples=disabled',
        'python=disabled',
        'sharp=disabled',
        'gtk=disabled',
        'libnice=enabled',
        'tls=enabled',
        'webrtc=enabled',
        'tools=disabled',
        'tests=disabled',
        'examples=disabled',
        'benchmarks=disabled',
        'extra-checks=disabled',
        'glib_assert=false',
        'glib_checks=false',
        'glib_debug=disabled',
        'introspection=disabled',
        'nls=disabled',
        'doc=disabled',
        'gtk_doc=disabled',
        'orc=enabled',
        'orc-source=subproject',
        'gstreamer:default_library=static',
        'gstreamer:benchmarks=disabled',
        'gstreamer:doc=disabled',
        'gstreamer:examples=disabled',
        'gstreamer:extra-checks=disabled',
        'gstreamer:glib_assert=false',
        'gstreamer:glib_checks=false',
        'gstreamer:glib_debug=disabled',
        'gstreamer:gst_debug=true',
        'gstreamer:gst_parse=false',
        'gstreamer:introspection=disabled',
        'gstreamer:nls=disabled',
        'gstreamer:registry=false',
        'gstreamer:tests=disabled',
        'gstreamer:tools=disabled',
        'gstreamer:tracer_hooks=false',
        'libnice:default_library=static',
        'libnice:examples=disabled',
        'libnice:glib_assert=false',
        'libnice:glib_checks=false',
        'libnice:glib_debug=disabled',
        'libnice:gtk_doc=disabled',
        'libnice:introspection=disabled',
        'libnice:tests=disabled',
        'gst-plugins-base:default_library=static',
        'gst-plugins-base:auto_features=disabled',
        'gst-plugins-base:app=enabled',
        'gst-plugins-base:doc=disabled',
        'gst-plugins-base:examples=disabled',
        'gst-plugins-base:glib_assert=false',
        'gst-plugins-base:glib_checks=false',
        'gst-plugins-base:glib_debug=disabled',
        'gst-plugins-base:introspection=disabled',
        'gst-plugins-base:jetstream-gst-plugins=app,rawparse,videoconvertscale',
        'gst-plugins-base:nls=disabled',
        'gst-plugins-base:tests=disabled',
        'gst-plugins-base:tools=disabled',
        'gst-plugins-base:rawparse=enabled',
        'gst-plugins-base:videoconvertscale=enabled',
        'gst-plugins-good:default_library=static',
        'gst-plugins-good:auto_features=disabled',
        'gst-plugins-good:doc=disabled',
        'gst-plugins-good:examples=disabled',
        'gst-plugins-good:glib_assert=false',
        'gst-plugins-good:glib_checks=false',
        'gst-plugins-good:glib_debug=disabled',
        'gst-plugins-good:jetstream-ext-plugins=vpx',
        'gst-plugins-good:jetstream-gst-plugins=rtp,rtpmanager',
        'gst-plugins-good:nls=disabled',
        'gst-plugins-good:qt5=disabled',
        'gst-plugins-good:qt6=disabled',
        'gst-plugins-good:rtp=enabled',
        'gst-plugins-good:rtpmanager=enabled',
        'gst-plugins-good:tests=disabled',
        'gst-plugins-good:vpx=enabled',
        'gst-plugins-bad:default_library=static',
        'gst-plugins-bad:auto_features=disabled',
        'gst-plugins-bad:debugutils=enabled',
        'gst-plugins-bad:doc=disabled',
        'gst-plugins-bad:dtls=enabled',
        'gst-plugins-bad:examples=disabled',
        'gst-plugins-bad:extra-checks=disabled',
        'gst-plugins-bad:glib_assert=false',
        'gst-plugins-bad:glib_checks=false',
        'gst-plugins-bad:glib_debug=disabled',
        'gst-plugins-bad:introspection=disabled',
        'gst-plugins-bad:jetstream-ext-plugins=dtls,openh264,sctp,srtp,webrtc',
        'gst-plugins-bad:jetstream-gst-plugins=videoparsers,debugutils',
        'gst-plugins-bad:nls=disabled',
        'gst-plugins-bad:openh264=enabled',
        'gst-plugins-bad:qroverlay=disabled',
        'gst-plugins-bad:sctp=enabled',
        'gst-plugins-bad:srtp=enabled',
        'gst-plugins-bad:tests=disabled',
        'gst-plugins-bad:videoparsers=enabled',
        'gst-plugins-bad:webrtc=enabled',
        'gst-plugins-rs:default_library=static',
        'gst-plugins-rs:auto_plugin_features=disabled',
        'gst-plugins-rs:doc=disabled',
        'gst-plugins-rs:examples=disabled',
        'gst-plugins-rs:rav1e=enabled',
        'gst-plugins-rs:rtp=enabled',
        'gst-plugins-rs:tests=disabled',
        'gst-plugins-rs:webrtc=enabled',
        'gst-plugins-rs:webrtc-aws=disabled',
        'gst-plugins-rs:webrtc-default-features=false',
        'gst-plugins-rs:webrtc-gcc=disabled',
        'gst-plugins-rs:webrtc-livekit=disabled',
        'gst-plugins-rs:webrtc-signalling-server=disabled',
        'gst-plugins-rs:webrtc-webrtcsrc=enabled',
        'gst-plugins-rs:webrtchttp=disabled',
    ]

    if jst_is_linux
        remote_gstreamer_options += [
            'gst-plugins-good:jetstream-sys-plugins=v4l2',
            'gst-plugins-good:v4l2=enabled',
        ]
    else
        remote_gstreamer_options += ['gst-plugins-good:v4l2=disabled']
    endif

    if jst_is_macos or jst_is_ios
        remote_gstreamer_options += [
            'gst-plugins-bad:applemedia=enabled',
            'gst-plugins-bad:jetstream-sys-plugins=applemedia',
        ]
    else
        remote_gstreamer_options += ['gst-plugins-bad:applemedia=disabled']
    endif

    if jst_is_linux or jst_is_windows
        remote_gstreamer_options += [
            'gst-plugins-bad:jetstream-sys-plugins=nvcodec',
            'gst-plugins-bad:nvcodec=enabled',
        ]
    else
        remote_gstreamer_options += ['gst-plugins-bad:nvcodec=disabled']
    endif

    gst_proj = subproject('gstreamer-full', required: remote_opt, default_options: remote_gstreamer_options)

    if gst_proj.found()
        deps = [
            gst_proj.get_variable('gst_full_dep'),
        ]

        if jst_is_macos or jst_is_ios
            deps += [
                dependency('appleframeworks', modules: ['SystemConfiguration'], required: remote_opt),
                cpp.find_library('iconv', required: remote_opt),
                cpp.find_library('resolv', required: remote_opt),
            ]

            if cpp.has_link_argument('-Wl,-no_compact_unwind')
                # Rust static plugins and C++ objects can exceed ld64's compact unwind personality table.
                deps += [declare_dependency(link_args: ['-Wl,-no_compact_unwind'])]
            endif
        endif
    else
        deps = [dependency('', required: false)]
    endif
else
    deps = []
endif

all_deps_found = remote_opt.allowed() and remote_cargo_c.found() and not jst_is_browser
foreach x_dep : deps
    all_deps_found = all_deps_found and x_dep.found()
endforeach

if all_deps_found
    cfg_lst.set('JETSTREAM_LOADER_GSTREAMER_AVAILABLE', true)
    dep_lst += deps
endif

ldr_lst += {'Gstreamer': all_deps_found}
