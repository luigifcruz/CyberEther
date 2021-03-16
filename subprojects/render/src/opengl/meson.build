gl_deps = [
  dependency('glfw3', required: false),
]

all_gl_deps_found = true
foreach x_dep : gl_deps
  all_gl_deps_found = all_gl_deps_found and x_dep.found()
endforeach

if all_gl_deps_found
  cfg_lst.set('RENDER_OPENGL_BACKEND_AVAILABLE', true)
  dep_lst += gl_deps
  src_lst += files([
    'opengl.cc',
  ])
  message('OpenGL Backend Available: YES')
else
  if get_option('OpenGL')
    error('OpenGL dependencies were not found.')
  endif
  message('OpenGL Backend Available: NO')
endif
