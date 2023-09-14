deps = [
    dependency('fftw3f', required: get_option('FFTW') or meson.is_cross_build()),
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and x_dep.found()
endforeach

if all_deps_found
    cfg_lst.set('JETSTREAM_LOADER_FFTW_AVAILABLE', true)
    dep_lst += deps
endif

ldr_lst += {'FFTW': all_deps_found}
