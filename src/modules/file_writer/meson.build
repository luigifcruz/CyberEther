src_lst += files([
])

backend_lst = []

subdir('cpu')
#subdir('cuda')

# TODO: Implement browser support.
if backend_lst.length() > 0 and not jst_is_browser
    cfg_lst.set('JETSTREAM_MODULE_FILE_WRITER_AVAILABLE', true)
    sum_lst += {'File Writer': backend_lst}
endif
