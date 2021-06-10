project(
    'render', 'cpp',
    version: '0.0.0',
    default_options: [
        'cpp_std=c++17',
    ]
)

cfg_lst = configuration_data()
src_lst = []
dep_lst = []
inc_lst = [
    include_directories(get_option('prefix') + '/include'),
    include_directories('include/render/tools'),
    include_directories('include'),
    include_directories('.'),
]

cuda_dep = dependency('cuda', version : '>=11', modules : ['cublas'])
if cuda_dep.found()
  cfg_lst.set('RENDER_CUDA_INTEROP_AVAILABLE', true)
  dep_lst += cuda_dep
else
  if get_option('CUDA')
    error('CUDA dependencies were not found.')
  endif
endif

summary({'cpp_std': get_option('cpp_std'),
         'prefix': get_option('prefix'),
         'buildtype': get_option('buildtype'),
         'CUDA interop': cuda_dep.found(),
        }, section: 'General', bool_yn: true)

subdir('src')
subdir('include')

librender = library(
    'render',
    src_lst,
    include_directories: inc_lst,
    dependencies: dep_lst,
    install: true
)

librender_dep = declare_dependency(
    include_directories: inc_lst,
    dependencies: dep_lst,
    link_with: librender
)
