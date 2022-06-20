cfg_lst.set('version', meson.project_version())
cfg_lst.set('build_type', get_option('prefix'))

ldr_lst = {}

subdir('cuda')
subdir('fftw')
subdir('metal')
subdir('glfw')

summary(ldr_lst, section: 'Loaders', bool_yn: true)
