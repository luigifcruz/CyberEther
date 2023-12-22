src_lst += files([
])

backend_lst = []

subdir('cpu')
subdir('metal')

if backend_lst.length() > 0
    cfg_lst.set('JETSTREAM_MODULE_MULTIPLY_CONSTANT_AVAILABLE', true)
    sum_lst += {'Multiply Constant': backend_lst}
endif
