if jst_is_browser
    node = find_program('node', native: true, required: true)
    test_link_args = browser_exe_link_args + ['-O0', '-g0']
else
    test_link_args = []
    add_test_setup('default', is_default: true, exclude_suites: ['sdr'])
endif

test_build_by_default = not jst_is_browser

subdir('hardware')

if catch2_dep.found()
    subdir('agent')
    subdir('core')
    subdir('modules')
endif

if py.found()
    test(
        'shader-parser',
        py,
        args : files('shaders/parser_test.py'),
        suite : 'shaders',
        timeout : 60,
    )
    geodata_test_python = pymod.find_installation(
        py.full_path(),
        required : true,
        modules : ['numpy', 'mapbox_earcut'],
    )
    test(
        'geodata-parser',
        geodata_test_python,
        args : files('geodata/parser_test.py'),
        suite : 'geodata',
        timeout : 60,
    )
endif
