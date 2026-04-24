project('naga', 'c', version: '22.1.0')

cargo = find_program('cargo')

run_command(cargo, 'build', '--release', '--bin', 'naga', check: true)

naga_path = meson.current_source_dir() + '/target/release/naga'
if build_machine.system() == 'windows'
  naga_path += '.exe'
endif

naga_bin = find_program(naga_path, required: true)
