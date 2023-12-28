cfg_lst.set('version', meson.project_version())
cfg_lst.set('build_type', get_option('prefix'))

ldr_lst = {}

subdir('cuda')
subdir('metal')
subdir('glfw')
subdir('soapy')
subdir('vulkan')
subdir('webgpu')
subdir('audiotoolbox')
subdir('gstreamer')
subdir('whispercpp')
# [NEW DEPENDENCY HOOK]

summary(ldr_lst, section: 'Loaders', bool_yn: true)
