deps = [
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and cfg_lst.get(x_dep, false)
endforeach

if all_deps_found
    cfg_lst.set('JETSTREAM_MODULE_FILTER_TAPS_AVAILABLE', true)
    cfg_lst.set('JETSTREAM_MODULE_FILTER_TAPS_CPU_AVAILABLE', true)
    
    src_lst += files([
        'generic.cc',
    ])

    sum_lst += {'Filter Taps': ['CPU']}
endif
