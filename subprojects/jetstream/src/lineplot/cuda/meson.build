cuda_deps = [
  dependency('cuda', version : '>=11', modules : ['cublas'], required: false)
]

all_cuda_deps_found = true
foreach x_dep : cuda_deps
  all_cuda_deps_found = all_cuda_deps_found and x_dep.found()
endforeach

if all_cuda_deps_found
  add_languages('cuda')
  cfg_lst.set('JETSTREAM_LPT_CUDA_AVAILABLE', true)
  dep_lst += cuda_deps
  src_lst += files([
    'base.cu',
  ])
else
  if get_option('CUDA')
    error('CUDA dependencies were not found.')
  endif
endif

sum_lst += {'GPU (CUDA)': all_cuda_deps_found}
