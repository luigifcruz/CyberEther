cuda_dep = dependency('cuda', version : '>=11', modules : ['cublas'], required: get_option('CUDA'))
if cuda_dep.found() and not meson.is_cross_build()
  cfg_lst.set('RENDER_CUDA_AVAILABLE', true)
  dep_lst += cuda_dep
endif
