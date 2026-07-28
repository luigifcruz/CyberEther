if jst_is_browser
    zlib_dep = declare_dependency(link_args: ['-sUSE_ZLIB=1'])
else
    zlib_dep = subproject(
        'zlib',
        required: true,
        default_options: [
            'default_library=static',
        ]
    ).get_variable('zlib_dep')
endif

deps = [zlib_dep]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and x_dep.found()
endforeach

if all_deps_found
    cfg_lst.set('JETSTREAM_LOADER_ZLIB_AVAILABLE', true)
    dep_lst += deps
endif

ldr_lst += {'zlib': all_deps_found}
