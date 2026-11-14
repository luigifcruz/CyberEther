project('ghostty-vt', 'c', version: '0.1.0', meson_version: '>=1.5.0')

zig = find_program(
    'zig',
    version: ['>=0.17.0', '<0.18.0'],
    version_argument: 'version',
    dirs: [
        '/opt/homebrew/opt/zig@0.17/bin',
        '/usr/local/opt/zig@0.17/bin',
        '/home/linuxbrew/.linuxbrew/opt/zig@0.17/bin',
    ],
    required: true,
)

zig_build_file = meson.current_source_dir() / 'build.zig'
zig_cpu = get_option('native') ? [] : ['-Dcpu=baseline']
zig_cache_dir = meson.current_build_dir() / 'zig-cache'

run_command(
    zig, 'build',
    '--build-file', zig_build_file,
    '--cache-dir', zig_cache_dir,
    '--fetch',
    '-Demit-lib-vt=true',
    check: true,
)

ghostty_vt_lib = custom_target(
    'ghostty-vt-build',
    output: ['libghostty-vt.a'],
    command: [
        zig, 'build',
        '--build-file', zig_build_file,
        '--cache-dir', zig_cache_dir,
        '--prefix', meson.current_build_dir() / 'zig-out',
        '--prefix-lib-dir', meson.current_build_dir(),
        '-Demit-lib-vt=true',
        '-Demit-xcframework=false',
        '-Doptimize=ReleaseFast',
    ] + zig_cpu,
    console: true,
    build_by_default: true,
)

ghostty_vt_dep = declare_dependency(
    include_directories: include_directories('include'),
    compile_args: ['-DGHOSTTY_STATIC'],
    link_with: ghostty_vt_lib,
)
