src_lst += files([
])

backend_lst = []

subdir('cpu')

if backend_lst.length() > 0
    cfg_lst.set('JETSTREAM_MODULE_FM_AVAILABLE', true)
    sum_lst += {'Frequency': backend_lst}
endif
