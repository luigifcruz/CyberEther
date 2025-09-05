src_lst += files([
])

backend_lst = []

subdir('cpu')

if backend_lst.length() > 0
    cfg_lst.set('JETSTREAM_MODULE_SIGNAL_GENERATOR_AVAILABLE', true)
    sum_lst += {'Signal Generator': backend_lst}
endif
