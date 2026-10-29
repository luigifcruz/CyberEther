src_lst += files([
    'generic.cc',
])

if cfg_lst.get('JETSTREAM_BACKEND_METAL_AVAILABLE', false)
    src_lst += files('metal.cc')
endif

sum_lst = {}

subdir('capture')
subdir('glfw')
subdir('headless')
subdir('ios')

some_viewport_found = false
foreach x_viewport : sum_lst.keys()
    some_viewport_found = some_viewport_found or sum_lst.get(x_viewport, false)
endforeach

summary(sum_lst, section: 'Viewport', bool_yn: true)
