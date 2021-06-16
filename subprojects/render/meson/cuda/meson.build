cuda_dep = dependency('cuda', version : '>=11', modules : ['cublas'], required: true)
if cuda_dep.found() and not meson.is_cross_build()
  cfg_lst.set('RENDER_CUDA_AVAILABLE', true)
  dep_lst += cuda_dep
else
  if get_option('CUDA')
    error('CUDA dependencies were not found.')
  endif
endif
