src_lst += files([
    'dialogs.cc',
    'dynamic_library.cc',
    'file.cc',
    'file_lock.cc',
    'http.cc',
    'paths.cc',
    'process.cc',
    'secret.cc',
    'storage.cc',
    'terminal.cc',
    'thread.cc',
    'url.cc',
    'websocket.cc',
])

if jst_is_browser or cfg_lst.get('JETSTREAM_LOADER_CPPHTTPLIB_AVAILABLE', false)
    cfg_lst.set('JETSTREAM_PLATFORM_HTTP_AVAILABLE', true)
endif

if jst_is_linux or jst_is_macos or jst_is_windows
    src_lst += files('interrupt.cc')
endif

if not jst_is_browser
    src_lst += files('socket.cc')
endif

if jst_is_ios
    dep_lst += [
        dependency('appleframeworks', modules: ['Foundation', 'UIKit', 'Security']),
    ]
endif

if jst_is_macos
    dep_lst += [
        dependency('appleframeworks', modules: ['Foundation', 'AppKit', 'UniformTypeIdentifiers', 'Security']),
    ]
endif

if jst_is_ios or jst_is_macos
    src_lst += files([
        'apple.mm',
    ])
endif

if jst_is_windows
    compiler = meson.get_compiler('cpp')
    dep_lst += [
        compiler.find_library('advapi32', required: true),
        compiler.find_library('comdlg32', required: true),
        compiler.find_library('iphlpapi', required: true),
        compiler.find_library('kernel32', required: true),
        compiler.find_library('shell32', required: true),
        compiler.find_library('ws2_32', required: true),
    ]
endif
