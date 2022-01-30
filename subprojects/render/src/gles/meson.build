gl_deps = [
  dependency('glfw3', required: false),
  dependency('glesv2', required: false),
]

all_gl_deps_found = true
foreach x_dep : gl_deps
  all_gl_deps_found = all_gl_deps_found and x_dep.found()
endforeach

if all_gl_deps_found
  cfg_lst.set('RENDER_GLES_AVAILABLE', true)
  dep_lst += gl_deps
  src_lst += files([
    'instance.cc',
    'surface.cc',
    'program.cc',
    'texture.cc',
    'vertex.cc',
    'draw.cc',
    'buffer.cc',
  ])

  subdir('tools')
else
  if get_option('OpenGL')
    error('OpenGL dependencies were not found.')
  endif
endif

sum_lst += {'OpenGL ES 3.0': all_gl_deps_found}
