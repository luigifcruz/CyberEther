project('nv-codec-headers', 'c', version: '12.2.72.0', license: 'MIT')

ffnvcodec_dep = declare_dependency(
  include_directories: include_directories('include'),
  version: meson.project_version(),
)
meson.override_dependency('ffnvcodec', ffnvcodec_dep)
