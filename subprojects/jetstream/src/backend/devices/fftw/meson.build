deps = [
    'JETSTREAM_LOADER_FFTW_AVAILABLE',
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and cfg_lst.get(x_dep, false)
endforeach

if all_deps_found
    cfg_lst.set('JETSTREAM_BACKEND_FFTW_AVAILABLE', true)
    src_lst += files([
    ])
endif

sum_lst += {'FFTW': all_deps_found}
