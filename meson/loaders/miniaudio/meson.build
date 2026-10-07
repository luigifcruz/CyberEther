all_deps_found = not (jst_is_ios or jst_is_macos) or \
                 cfg_lst.get('JETSTREAM_LOADER_AUDIOTOOLBOX_AVAILABLE', false)

if all_deps_found
    cfg_lst.set('JETSTREAM_LOADER_MINIAUDIO_AVAILABLE', true)
endif

ldr_lst += {'miniaudio': all_deps_found}
