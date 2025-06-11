src_lst += files([
    'generic.cc',
])

sum_lst = {}

subdir('glfw')
subdir('headless')
subdir('ios')

some_viewport_found = false
foreach x_viewport : sum_lst.keys()
    some_viewport_found = some_viewport_found or sum_lst.get(x_viewport, false)
endforeach

summary(sum_lst, section: 'Viewport', bool_yn: true)
