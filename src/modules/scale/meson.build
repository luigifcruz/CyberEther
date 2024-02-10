src_lst += files([
])

backend_lst = []

subdir('cpu')
subdir('metal')
subdir('cuda')

if backend_lst.length() > 0
    cfg_lst.set('JETSTREAM_MODULE_SCALE_AVAILABLE', true)
    sum_lst += {'Scale': backend_lst}
endif
