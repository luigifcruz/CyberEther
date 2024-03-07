src_lst += files([
])

backend_lst = []

subdir('cpu')
subdir('cuda')

if backend_lst.length() > 0
    cfg_lst.set('JETSTREAM_MODULE_ARITHMETIC_AVAILABLE', true)
    sum_lst += {'Arithmetic': backend_lst}
endif
