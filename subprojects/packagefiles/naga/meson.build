project('naga', 'c', version: '0.13.0')

cargo = find_program('cargo')

run_command(cargo, 'build', '--release', check: true)

naga_bin = find_program(meson.current_source_dir() + '/target/release/naga', required: true)