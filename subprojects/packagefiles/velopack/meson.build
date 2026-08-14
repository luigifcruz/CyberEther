project('velopack', 'cpp', version: '1.2.0')

system = target_machine.system()
cpu = target_machine.cpu_family()

if system == 'windows'
    if cpu == 'x86_64'
        velopack_link_name = 'velopack_libc_win_x64_msvc.lib'
    elif cpu == 'aarch64'
        velopack_link_name = 'velopack_libc_win_arm64_msvc.lib'
    else
        error('Velopack does not support Windows architecture: ' + cpu)
    endif
elif system == 'darwin'
    if cpu == 'aarch64'
        velopack_link_name = 'velopack_libc_osx_arm64_gnu.a'
    elif cpu == 'x86_64'
        velopack_link_name = 'velopack_libc_osx_x64_gnu.a'
    else
        error('Velopack does not support macOS architecture: ' + cpu)
    endif
elif system == 'linux'
    if cpu == 'x86_64'
        velopack_link_name = 'velopack_libc_linux_x64_gnu.a'
    elif cpu == 'aarch64'
        velopack_link_name = 'velopack_libc_linux_arm64_gnu.a'
    else
        error('Velopack does not support Linux architecture: ' + cpu)
    endif
else
    error('Velopack does not support target system: ' + system)
endif

velopack_include = include_directories('include')
velopack_link_path = meson.current_source_dir() / 'lib-static' / velopack_link_name
