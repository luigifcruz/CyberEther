all_deps_found = jst_is_ios or jst_is_macos

if all_deps_found
    add_languages('objcpp', native: false)
    objcpp = meson.get_compiler('objcpp')
    if objcpp.has_link_argument('-nostdlib++')
        add_project_link_arguments('-nostdlib++', language: 'objcpp')
    endif
endif

ldr_lst += {'Objective-C++': all_deps_found}
