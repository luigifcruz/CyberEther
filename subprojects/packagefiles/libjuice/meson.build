project('libjuice', 'c', version: '1.7.5', license: 'MPL-2.0', default_options: ['c_std=gnu11,c11', 'optimization=2'])

cc = meson.get_compiler('c')
is_windows = host_machine.system() == 'windows'

deps = [dependency('threads')]
if is_windows
  deps += [cc.find_library('ws2_32'), cc.find_library('bcrypt')]
endif

args = [
  '-DJUICE_EXPORTS',
  '-DJUICE_STATIC',
  '-DUSE_NETTLE=0',
  '-DRELEASE=1',
]
if is_windows
  args += ['-DWIN32_LEAN_AND_MEAN', '-DNOMINMAX', '-D_CRT_SECURE_NO_WARNINGS']
endif

sources = files(
  'src/addr.c',
  'src/agent.c',
  'src/base64.c',
  'src/conn.c',
  'src/conn_mux.c',
  'src/conn_poll.c',
  'src/conn_thread.c',
  'src/const_time.c',
  'src/crc32.c',
  'src/hash.c',
  'src/hmac.c',
  'src/ice.c',
  'src/juice.c',
  'src/log.c',
  'src/random.c',
  'src/server.c',
  'src/stun.c',
  'src/tcp.c',
  'src/timestamp.c',
  'src/turn.c',
  'src/udp.c',
)

libjuice = static_library('juice', sources,
  c_args: args + cc.get_supported_arguments(['-Wno-unused-parameter', '-Wno-sign-compare']),
  include_directories: include_directories('include', 'include/juice', 'src'),
  dependencies: deps,
  pic: true,
)

libjuice_dep = declare_dependency(
  compile_args: ['-DJUICE_STATIC'],
  include_directories: include_directories('include'),
  link_with: libjuice,
  dependencies: deps,
)
meson.override_dependency('libjuice', libjuice_dep)
