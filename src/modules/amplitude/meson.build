src_lst += files([
])

backend_lst = []

subdir('cpu')
subdir('metal')

if backend_lst.length() > 0
    cfg_lst.set('JETSTREAM_MODULE_AMPLITUDE_AVAILABLE', true)
    sum_lst += {'Amplitude': backend_lst}
endif
