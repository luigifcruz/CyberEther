version_cfg = configuration_data()
version_cfg.set('VA_API_MAJOR_VERSION', 1)
version_cfg.set('VA_API_MINOR_VERSION', 24)
version_cfg.set('VA_API_MICRO_VERSION', 0)
version_cfg.set('VA_API_VERSION', '1.24.0')

version_file = configure_file(
    input: 'va_version.h.in',
    output: 'va_version.h',
    configuration: version_cfg,
)

drm_header = configure_file(
    input: 'drm/va_drm.h',
    output: 'va_drm.h',
    copy: true,
)

va_inc = include_directories('..')

libva = static_library(
    'va',
    ['va.c', 'va_compat.c', 'va_str.c', 'va_trace.c', version_file],
    c_args: va_args,
    include_directories: va_inc,
    dependencies: [dl_dep],
    gnu_symbol_visibility: 'hidden',
)

libva_dep = declare_dependency(
    link_with: libva,
    include_directories: va_inc,
    sources: [version_file],
    dependencies: [dl_dep],
    version: '1.24.0',
)

libva_drm = static_library(
    'va-drm',
    ['drm/va_drm.c', 'drm/va_drm_auth.c', 'drm/va_drm_utils.c', version_file, drm_header],
    c_args: va_args,
    include_directories: va_inc,
    dependencies: [libdrm_dep],
    gnu_symbol_visibility: 'hidden',
)

libva_drm_dep = declare_dependency(
    link_with: libva_drm,
    include_directories: va_inc,
    sources: [drm_header],
    dependencies: [libva_dep, libdrm_dep],
    version: '1.24.0',
)

meson.override_dependency('libva', libva_dep)
meson.override_dependency('libva-drm', libva_drm_dep)
