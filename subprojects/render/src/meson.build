src_lst += files([
  'base.cc',
  'type.cc',
])

sum_lst = {}

subdir('gles')
subdir('metal')

summary(sum_lst, section: 'Render Backends', bool_yn: true)

subdir('tools')
