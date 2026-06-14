if not jst_is_ios and \
   not jst_is_android and \
   not jst_is_browser
    catch2_dep = subproject(
        'catch2',
        required: true,
        default_options: [
            'default_library=static',
            'tests=false',
            'cpp_std=c++20',
        ]
    ).get_variable('catch2_dep')
    tools_lst += {'Catch2': catch2_dep.found()}
endif
