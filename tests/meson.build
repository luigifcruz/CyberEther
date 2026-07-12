if catch2_dep.found()
    subdir('memory')
    subdir('platform')
    subdir('parser')
    subdir('settings')
    subdir('modules')
    subdir('flowgraph')
endif

if catch2_dep.found() and not jst_is_ios and not jst_is_android
    subdir('cli')
endif

subdir('components')
