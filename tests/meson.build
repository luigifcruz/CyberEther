if catch2_dep.found()
    subdir('memory')
    subdir('platform')
    subdir('parser')
    subdir('settings')
    subdir('modules')
    subdir('flowgraph')
    subdir('render')
endif

# TODO: Re-enable after CLI output capture supports separate Windows static CRTs.
if catch2_dep.found() and \
   not jst_is_ios and \
   not jst_is_android and \
   not jst_is_windows
    subdir('cli')
endif

subdir('components')
