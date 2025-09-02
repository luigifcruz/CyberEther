deps = []

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and cfg_lst.get(x_dep, false)
endforeach

if all_deps_found
    backend_lst += 'CPU'
    cfg_lst.set('JETSTREAM_MODULE_FILE_READER_CPU_AVAILABLE', true)
    src_lst += files([
        'base.cc',
    ])
endif
