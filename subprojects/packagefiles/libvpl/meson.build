project('libvpl', 'cpp', version: '2.17.0', license: 'MIT', default_options: ['cpp_std=c++17', 'warning_level=0'])

if host_machine.system() != 'windows'
  error('The libvpl subproject only builds the Windows dispatcher.')
endif

cpp = meson.get_compiler('cpp')

deps = [
  dependency('threads'),
  cpp.find_library('advapi32'),
  cpp.find_library('cfgmgr32'),
]

api_inc = include_directories('api', 'api/vpl')

sources = files(
  'libvpl/src/mfx_config_interface/mfx_config_interface.cpp',
  'libvpl/src/mfx_config_interface/mfx_config_interface_string_api.cpp',
  'libvpl/src/mfx_dispatcher_vpl.cpp',
  'libvpl/src/mfx_dispatcher_vpl_config.cpp',
  'libvpl/src/mfx_dispatcher_vpl_loader.cpp',
  'libvpl/src/mfx_dispatcher_vpl_log.cpp',
  'libvpl/src/mfx_dispatcher_vpl_lowlatency.cpp',
  'libvpl/src/mfx_dispatcher_vpl_msdk.cpp',
  'libvpl/src/windows/main.cpp',
  'libvpl/src/windows/mfx_critical_section.cpp',
  'libvpl/src/windows/mfx_dispatcher.cpp',
  'libvpl/src/windows/mfx_dispatcher_log.cpp',
  'libvpl/src/windows/mfx_driver_store_loader.cpp',
  'libvpl/src/windows/mfx_dxva2_device.cpp',
  'libvpl/src/windows/mfx_function_table.cpp',
  'libvpl/src/windows/mfx_library_iterator.cpp',
  'libvpl/src/windows/mfx_load_dll.cpp',
  'libvpl/src/windows/mfx_win_reg_key.cpp',
)

libvpl = static_library('vpl',
  sources,
  cpp_args: ['-DMFX_DEPRECATED_OFF'],
  include_directories: [api_inc, include_directories('libvpl')],
  dependencies: deps,
  gnu_symbol_visibility: 'hidden',
)

vpl_dep = declare_dependency(
  link_with: libvpl,
  include_directories: api_inc,
  compile_args: ['-DMFX_DEPRECATED_OFF'],
  dependencies: deps,
  version: meson.project_version(),
)
meson.override_dependency('vpl', vpl_dep)
