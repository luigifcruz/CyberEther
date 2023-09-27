if target_machine.system() == 'ios' or target_machine.system() == 'darwin'
    src_lst += files([
        'miniaudio.mm',
    ])
else
    src_lst += files([
        'miniaudio.cc',
    ])
endif

sum_lst = {}

subdir('cpu')
subdir('metal')

summary(sum_lst, section: 'Audio Module', bool_yn: true)
