deps = [
    'JETSTREAM_LOADER_SOAPY_AVAILABLE',
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and cfg_lst.get(x_dep, false)
endforeach

if all_deps_found
    cfg_lst.set('JETSTREAM_MODULE_SOAPY_CPU_AVAILABLE', true)
    src_lst += files([
        'base.cc',
    ])
endif

backend_lst += 'CPU'
