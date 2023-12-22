src_lst += files([
])

backend_lst = []

subdir('cpu')

if backend_lst.length() > 0
    cfg_lst.set('JETSTREAM_MODULE_SPEECH_RECOGNITION_AVAILABLE', true)
    sum_lst += {'Speech Recognition': backend_lst}
endif
