cuda_deps = [
  dependency('cuda', required: false),
  dependency('cufft', required: false),
]

all_cuda_deps_found = true
foreach x_dep : cuda_deps
  all_cuda_deps_found = all_cuda_deps_found and x_dep.found()
endforeach

if all_cuda_deps_found
  cfg_lst.set('JETSTREAM_FFT_CUDA_AVAILABLE', true)
  dep_lst += cuda_deps
  src_lst += files([
    'base.cc',
  ])
  message('CUDA Backend Available: YES')
else
  if get_option('CUDA')
    error('CUDA dependencies were not found.')
  endif
  message('CUDA Backend Available: NO')
endif
