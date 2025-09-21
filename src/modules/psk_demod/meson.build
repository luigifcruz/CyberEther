src_lst += files([
])

backend_lst = []

subdir('cpu')

if backend_lst.length() > 0
    cfg_lst.set('JETSTREAM_MODULE_PSK_DEMOD_AVAILABLE', true)
    sum_lst += {'PSK Demod': backend_lst}
endif
