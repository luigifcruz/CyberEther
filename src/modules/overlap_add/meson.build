src_lst += files([
])

backend_lst = []

subdir('cpu')

if backend_lst.length() > 0
    cfg_lst.set('JETSTREAM_MODULE_OVERLAP_ADD_AVAILABLE', true)
    sum_lst += {'Overlap Add': backend_lst}
endif
