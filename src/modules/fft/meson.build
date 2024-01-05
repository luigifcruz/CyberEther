src_lst += files([
])

backend_lst = []

subdir('cpu')
subdir('metal')

if backend_lst.length() > 0
    cfg_lst.set('JETSTREAM_MODULE_FFT_AVAILABLE', true)
    sum_lst += {'FFT': backend_lst}
endif
