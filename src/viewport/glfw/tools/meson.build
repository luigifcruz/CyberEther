src_lst += files([
    'glfw_input.cc',
])

if jst_is_browser
    src_lst += files([
        'browser_wheel.cc',
    ])
endif

if jst_is_macos
    src_lst += files([
        'cocoa_pointer.mm',
        'cocoa_titlebar.mm',
    ])
endif

if jst_is_windows
    src_lst += files('windows_titlebar.cc', 'windows_pointer.cc')
    dep_lst += [cpp.find_library('dwmapi'), cpp.find_library('comctl32'), cpp.find_library('uxtheme'),
                cpp.find_library('ole32')]
endif
