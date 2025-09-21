src_lst += files([
])

backend_lst = []

subdir('cpu')

if backend_lst.length() > 0
    cfg_lst.set('JETSTREAM_MODULE_RRC_FILTER_AVAILABLE', true)
    sum_lst += {'RRC Filter': backend_lst}
endif
