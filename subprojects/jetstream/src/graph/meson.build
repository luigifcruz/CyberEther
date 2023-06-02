src_lst += files([
    'generic.cc',
])

sum_lst = {}

subdir('cpu')
subdir('metal')

summary(sum_lst, section: 'Graph Backend', bool_yn: true)
