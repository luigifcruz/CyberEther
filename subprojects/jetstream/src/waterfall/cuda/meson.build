cuda_deps = [
  dependency('cuda', version : '>=10', modules : ['cublas'], required: get_option('CUDA'))
]

all_cuda_deps_found = true
foreach x_dep : cuda_deps
  all_cuda_deps_found = all_cuda_deps_found and x_dep.found()
endforeach
all_cuda_deps_found = all_cuda_deps_found and not meson.is_cross_build()

if all_cuda_deps_found
  add_languages('cuda')
  cfg_lst.set('JETSTREAM_WTF_CUDA_AVAILABLE', true)
  dep_lst += cuda_deps
  src_lst += files([
    'base.cu',
  ])
endif

sum_lst += {'GPU (CUDA)': all_cuda_deps_found}

