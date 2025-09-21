src_lst += files([
])

backend_lst = []

subdir('cpu')

if backend_lst.length() > 0
    cfg_lst.set('JETSTREAM_MODULE_FILE_READER_AVAILABLE', true)
    sum_lst += {'File Reader': backend_lst}
endif
