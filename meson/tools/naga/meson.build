if jst_is_browser
    naga_bin = subproject('naga', required: true).get_variable('naga_bin')
else
    naga_bin = find_program('naga', required: false)
endif

tools_lst += {'Naga': naga_bin.found()}
