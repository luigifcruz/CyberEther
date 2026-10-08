src_lst += files([
    'circular_buffer.cc',
    'json.cc',
])

if cfg_lst.get('JETSTREAM_LOADER_MINIAUDIO_AVAILABLE', false)
    if jst_is_ios or jst_is_macos
        src_lst += files('miniaudio/miniaudio.mm')
    else
        src_lst += files('miniaudio/miniaudio.cc')
    endif
endif
