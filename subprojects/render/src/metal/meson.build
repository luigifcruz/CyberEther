metal_fmrk = [
  'MetalKit',
  'Metal',
  'Foundation',
  'QuartzCore',
  'Cocoa',
  'GameController',
]

metal_deps = [
  dependency('appleframeworks', modules: metal_fmrk, required: false),
  dependency('glfw3', required: false),
]

all_metal_deps_found = true
foreach x_dep : metal_deps
  all_metal_deps_found = all_metal_deps_found and x_dep.found()
endforeach

if all_metal_deps_found
  add_languages('objcpp')

  cfg_lst.set('RENDER_METAL_AVAILABLE', true)
  dep_lst += metal_deps
  src_lst += files([
    'instance.cc',
    'surface.cc',
    'program.cc',
    'texture.cc',
    'vertex.cc',
    'draw.cc',
    'window.mm',
  ])

  subdir('tools')
else
  if get_option('Metal')
    error('Metal dependencies were not found.')
  endif
endif

sum_lst += {'Metal': all_metal_deps_found}
