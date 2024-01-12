if not jst_is_ios and \
   not jst_is_android and \
   not jst_is_browser
    catch2_dep = dependency(
        'catch2-with-main', 
        fallback: 'catch2', 
        default_options: [
            'default_library=static',
            'tests=false',
        ]
    )
    tools_lst += {'Catch2': catch2_dep.found()}
endif