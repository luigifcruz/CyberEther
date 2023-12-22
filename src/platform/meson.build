src_lst += files([
    'base.cc',
])

if target_machine.system() == 'ios'
    dep_lst += [
        dependency('appleframeworks', modules: ['Foundation', 'UIKit']),
    ]
endif

if target_machine.system() == 'darwin'
    dep_lst += [
        dependency('appleframeworks', modules: ['Foundation', 'AppKit']),
    ]
endif

if target_machine.system() == 'ios' or target_machine.system() == 'darwin'
    src_lst += files([
        'apple.mm',
    ])
endif