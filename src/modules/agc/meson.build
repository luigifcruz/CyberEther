src_lst += files([
])

backend_lst = []

subdir('cpu')

if backend_lst.length() > 0
    cfg_lst.set('JETSTREAM_MODULE_AGC_AVAILABLE', true)
    sum_lst += {'AGC': backend_lst}
endif
