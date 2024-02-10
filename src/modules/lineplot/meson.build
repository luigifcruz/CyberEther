src_lst += files([
])

backend_lst = []

subdir('cpu')
subdir('metal')
subdir('cuda')

if backend_lst.length() > 0
    cfg_lst.set('JETSTREAM_MODULE_LINEPLOT_AVAILABLE', true)
    sum_lst += {'Lineplot': backend_lst}
endif