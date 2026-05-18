package_macos = find_program('package.sh')
package_env = find_program('env')
dmg_name = 'CyberEther-' + meson.project_version() + '.dmg'

custom_target(
    'cyberether-dmg',
    input: cyberether_exe,
    output: dmg_name,
    command: [
        package_env,
        'CYBERETHER_BINARY=@INPUT@',
        'OUTPUT_DIR=' + meson.current_build_dir(),
        'DMG_PATH=' + join_paths(meson.current_build_dir(), dmg_name),
        'VERSION=' + meson.project_version(),
        package_macos,
    ],
    build_by_default: false,
)
