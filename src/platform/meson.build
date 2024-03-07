src_lst += files([
    'base.cc',
])

if jst_is_ios
    dep_lst += [
        dependency('appleframeworks', modules: ['Foundation', 'UIKit']),
    ]
endif

if jst_is_macos
    dep_lst += [
        dependency('appleframeworks', modules: ['Foundation', 'AppKit', 'UniformTypeIdentifiers']),
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
        compiler.find_library('comdlg32', required: true),
        compiler.find_library('kernel32', required: true),
        compiler.find_library('shell32', required: true),
    ]
endif