if catch2_dep.found() and module_test_lst.length() > 0
    test('modules', executable(
        'jetstream-modules-tests',
        ['main.cc', 'runtime_skip.cc'] + module_test_lst,
        dependencies: [libjetstream_dep, catch2_dep],
        include_directories: include_directories('../common'),
    ), is_parallel: false, timeout: 0)
endif
