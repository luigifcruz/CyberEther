deps = [
    'JETSTREAM_BACKEND_METAL_AVAILABLE',
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and cfg_lst.get(x_dep, false)
endforeach
all_deps_found = all_deps_found and jst_is_ios

if all_deps_found
    cfg_lst.set('JETSTREAM_VIEWPORT_IOS_AVAILABLE', true)
    src_lst += files([
        'metal.cc',
    ])
endif

sum_lst += {'iOS': all_deps_found}