terminal_opt = get_option('terminal')
terminal_supported = (jst_is_linux or jst_is_macos) and not jst_is_crossbuild and not jst_is_wheel

ghostty_vt_found = false
if terminal_supported and not terminal_opt.disabled()
    ghostty_vt_project = subproject('ghostty-vt', required: terminal_opt)
    if ghostty_vt_project.found()
        dep_lst += [ghostty_vt_project.get_variable('ghostty_vt_dep')]
        cfg_lst.set('JETSTREAM_LOADER_GHOSTTY_VT_AVAILABLE', true)
        ghostty_vt_found = true
    endif
elif terminal_opt.enabled()
    error('The integrated terminal is only available on native Linux and macOS builds.')
endif

ldr_lst += {'Ghostty VT': ghostty_vt_found}
