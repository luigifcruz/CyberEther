pocketfft_subproject = subproject('pocketfft', required: true)
pocketfft_dep = pocketfft_subproject.get_variable('pocketfft_dep')

dep_lst += [pocketfft_dep]
cfg_lst.set('JETSTREAM_LOADER_POCKETFFT_AVAILABLE', true)
ldr_lst += {'PocketFFT': pocketfft_dep.found()}
