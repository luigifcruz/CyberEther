if not jst_is_ios and \
   not jst_is_android and \
   not jst_is_browser
    catch2_dep = dependency(
        'catch2', 
        fallback: 'catch2', 
        version: '>=3.0.0', 
        required: false, 
        default_options: [
            'default_library=static',
            'tests=false',
            'cpp_std=c++20',
        ]
    )
    tools_lst += {'Catch2': catch2_dep.found()}
endif