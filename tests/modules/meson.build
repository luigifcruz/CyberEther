if catch2_dep.found() and module_test_lst.length() > 0
    module_test_fs = import('fs')
    module_test_dependencies = [libjetstream_dep, catch2_dep]
    module_test_main = static_library(
        'jetstream-modules-test-main',
        'main.cc',
        dependencies: module_test_dependencies,
    )
    module_test_targets = []

    foreach test_source : module_test_lst
        test_component_path = module_test_fs.parent(test_source)
        test_domain = module_test_fs.name(module_test_fs.parent(test_component_path))
        test_component = module_test_fs.name(test_component_path)
        test_filename = module_test_fs.name(test_source)
        if test_filename == 'module_tests.cc'
            test_kind = 'module'
        elif test_filename == 'block_tests.cc'
            test_kind = 'block'
        else
            error('Unsupported module test source: ' + test_filename)
        endif
        test_name = test_domain + '-' + test_component + '-' + test_kind

        test_target = executable(
            'jetstream-modules-' + test_name + '-tests',
            test_source,
            dependencies: module_test_dependencies,
            link_with: module_test_main,
            include_directories: include_directories('../support'),
        )
        module_test_targets += [test_target]

        test(
            'modules-' + test_name,
            test_target,
            suite: 'modules',
            is_parallel: false,
            timeout: 180,
        )
    endforeach

    alias_target(
        'jetstream-modules-tests',
        module_test_targets,
    )
endif
