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

if jst_is_linux
    add_languages('c', native: false)
    wayland_client = dependency('wayland-client', version: '>=1.21')
    dep_lst += wayland_client.partial_dependency(compile_args: true, includes: true)
    pointer_protocols = dependency('wayland-protocols', version: '>=1.23')
    pointer_scanner = find_program('wayland-scanner', native: true)
    pointer_xml = pointer_protocols.get_variable(pkgconfig: 'pkgdatadir') / 'unstable/pointer-gestures/pointer-gestures-unstable-v1.xml'
    wayland_scanner = dependency('wayland-scanner', native: true)
    wayland_xml = wayland_scanner.get_variable(pkgconfig: 'pkgdatadir') / 'wayland.xml'
    wayland_pointer_protocols = [
        custom_target('wayland-code', input: wayland_xml, output: 'wayland-code.c',
                      command: [pointer_scanner, 'private-code', '@INPUT@', '@OUTPUT@']),
        custom_target('pointer-gestures-client', input: pointer_xml, output: 'pointer-gestures-client.h',
                      command: [pointer_scanner, 'client-header', '@INPUT@', '@OUTPUT@']),
        custom_target('pointer-gestures-code', input: pointer_xml, output: 'pointer-gestures-code.c',
                      command: [pointer_scanner, 'private-code', '@INPUT@', '@OUTPUT@']),
    ]
    src_lst += files('wayland_pointer.cc') + wayland_pointer_protocols
    inc_lst += include_directories('.')
endif
