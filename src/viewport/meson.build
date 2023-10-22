src_lst += files([
    'generic.cc',
])

sum_lst = {}

subdir('glfw')
subdir('ios')
subdir('headless')

subdir('plugins')

some_viewport_found = false
foreach x_viewport : sum_lst.keys()
    some_viewport_found = some_viewport_found or sum_lst.get(x_viewport, false)
endforeach

if not some_viewport_found
    error('Failed to find a compatible viewport. Check your dependencies and try again.')
endif

summary(sum_lst, section: 'Viewport', bool_yn: true)
