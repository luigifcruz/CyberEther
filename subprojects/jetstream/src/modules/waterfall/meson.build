src_lst += files([
  'generic.cc',
])

sum_lst = {}

subdir('cpu')
subdir('cuda')

summary(sum_lst, section: 'Waterfall Module', bool_yn: true)
