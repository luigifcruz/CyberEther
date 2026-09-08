if catch2_dep.found()
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
