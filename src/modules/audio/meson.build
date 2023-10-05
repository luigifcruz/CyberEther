if target_machine.system() == 'ios' or target_machine.system() == 'darwin'
    src_lst += files([
        'miniaudio.mm',
    ])
else
    src_lst += files([
        'miniaudio.cc',
    ])
endif

backend_lst = []

subdir('cpu')
subdir('metal')

sum_lst += {'Audio': backend_lst}
