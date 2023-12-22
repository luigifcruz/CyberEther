src_lst += files([
])

backend_lst = []

subdir('cpu')

if backend_lst.length() > 0
    cfg_lst.set('JETSTREAM_MODULE_INVERT_AVAILABLE', true)
    sum_lst += {'Invert': backend_lst}
endif
