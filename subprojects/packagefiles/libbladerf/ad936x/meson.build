ad936x_project = subproject('bladerf-no-os')
fs = import('fs')
ad936x_inc = include_directories('.')
ad936x_sources = []
foreach source : ad936x_project.get_variable('ad936x_files')
  ad936x_sources += configure_file(
    input: source,
    output: fs.name(source),
    copy: true,
  )
endforeach

# Keep these configure-time files separate from the former build-time outputs
# so Ninja's stale-output cleanup cannot remove the newly generated headers.
# The ceiling keeps Git from treating an in-tree build as repository files.
git = find_program('git')
patch_env = environment()
patch_env.set('GIT_CEILING_DIRECTORIES', fs.parent(meson.current_build_dir()))
foreach patch_file : files(
  '../thirdparty/analogdevicesinc/no-OS_local/patches/0001-0bba46e-nuand-modifications.patch',
  '../thirdparty/analogdevicesinc/no-OS_local/patches/0002-0bba46e-pr-561.patch',
  '../thirdparty/analogdevicesinc/no-OS_local/patches/0003-0bba46e-pr-573.patch',
  '../thirdparty/analogdevicesinc/no-OS_local/patches/0004-0bba46e-pr-598.patch',
  '../thirdparty/analogdevicesinc/no-OS_local/patches/0005-0bba46e-rm-ad9361_parse_fir.patch',
  '../thirdparty/analogdevicesinc/no-OS_local/patches/0006-0bba46e-compilefix.patch',
)
  run_command(
    git, '-C', meson.current_build_dir(), 'apply',
    '--whitespace=nowarn', '-p3', patch_file,
    env: patch_env,
    check: true,
  )
endforeach
