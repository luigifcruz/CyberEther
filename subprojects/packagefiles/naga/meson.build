project('naga', 'c', version: '22.1.0')

cargo = find_program('cargo')

run_command(cargo, 'build', '--release', '--bin', 'naga', check: true)

naga_bin = find_program(meson.current_source_dir() + '/target/release/naga', required: true)