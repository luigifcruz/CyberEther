project('amf-headers', 'c', version: '1.5.3', license: 'MIT')

amf_dep = declare_dependency(
  include_directories: include_directories('.'),
  version: meson.project_version(),
)
meson.override_dependency('amf', amf_dep)
