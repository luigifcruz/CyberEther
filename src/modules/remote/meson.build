backend_lst = []

deps = [
    'JETSTREAM_LOADER_GSTREAMER_AVAILABLE',
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and cfg_lst.get(x_dep, false)
endforeach

if all_deps_found
    cfg_lst.set('JETSTREAM_MODULE_REMOTE_CPU_AVAILABLE', true)
    src_lst += files([
        'generic.cc',
    ])
endif

backend_lst += 'CPU'

sum_lst += {'Remote': backend_lst}
