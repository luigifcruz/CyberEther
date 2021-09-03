fftw_deps = [
  dependency('fftw3f', required: false),
]

all_fftw_deps_found = true
foreach x_dep : fftw_deps
  all_fftw_deps_found = all_fftw_deps_found and x_dep.found()
endforeach

if all_fftw_deps_found
  cfg_lst.set('JETSTREAM_FFT_FFTW_AVAILABLE', true)
  dep_lst += fftw_deps
  src_lst += files([
    'base.cc',
  ])
else
  if get_option('FFTW')
    error('FFTW dependencies were not found.')
  endif
endif

sum_lst += {'CPU (FFTW)': all_fftw_deps_found}
