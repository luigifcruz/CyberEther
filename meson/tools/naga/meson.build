naga_root = subproject('naga', required: false)
if naga_root.found()
naga_bin = naga_root.get_variable('naga_bin')
else
naga_bin = naga_root
endif
tools_lst += {'Naga': naga_bin.found()}