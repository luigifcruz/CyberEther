deps = [
    dependency('fftw3f', required: get_option('FFTW')),
    dependency('fftw3', required: get_option('FFTW')),
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and x_dep.found()
endforeach
all_deps_found = all_deps_found and not meson.is_cross_build()

if all_deps_found
    cfg_lst.set('JETSTREAM_FFTW_AVAILABLE', true)
    dep_lst += deps
endif

ldr_lst += {'FFTW': all_deps_found}
