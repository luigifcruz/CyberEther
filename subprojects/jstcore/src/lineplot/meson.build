src_lst += files([
  'generic.cc',
])

dep_lst += [
  subproject('render').get_variable('librender_dep'),
]

sum_lst = {}

subdir('cpu')
subdir('cuda')

summary(sum_lst, section: 'Lineplot Module', bool_yn: true)
