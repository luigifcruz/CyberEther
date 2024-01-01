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
        dependency('appleframeworks', modules: ['Foundation', 'AppKit']),
    ]
endif

if jst_is_ios or jst_is_macos
    src_lst += files([
        'apple.mm',
    ])
endif