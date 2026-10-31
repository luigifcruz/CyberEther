project('libzmq', 'cpp', 'c', version: '4.3.5', license: 'MPL-2.0', default_options: ['cpp_std=c++11', 'optimization=2'])

cxx = meson.get_compiler('cpp')
system = host_machine.system()
is_windows = system == 'windows'
is_linux = system == 'linux'
is_darwin = system == 'darwin'

conf = configuration_data()
conf.set('ZMQ_USE_CV_IMPL_STL11', 1)
conf.set('ZMQ_CACHELINE_SIZE', 64)
conf.set('ZMQ_HAVE_NOEXCEPT', 1)

if is_windows
  conf.set('ZMQ_HAVE_WINDOWS', 1)
  conf.set('ZMQ_IOTHREAD_POLLER_USE_EPOLL', 1)
  conf.set('ZMQ_POLL_BASED_ON_POLL', 1)
  conf.set('HAVE_IF_NAMETOINDEX', 1)
  conf.set('HAVE_STRNLEN', 1)
elif is_darwin
  conf.set('ZMQ_IOTHREAD_POLLER_USE_KQUEUE', 1)
  conf.set('ZMQ_POLL_BASED_ON_POLL', 1)
else
  conf.set('ZMQ_IOTHREAD_POLLER_USE_EPOLL', 1)
  conf.set('ZMQ_POLL_BASED_ON_POLL', 1)
  if cxx.has_header_symbol('sys/epoll.h', 'epoll_create1')
    conf.set('ZMQ_IOTHREAD_POLLER_USE_EPOLL_CLOEXEC', 1)
  endif
endif

if not is_windows
  gnu = is_linux ? ['-D_GNU_SOURCE'] : []
  checks = {
    'HAVE_POSIX_MEMALIGN': ['stdlib.h', 'posix_memalign'],
    'HAVE_FORK': ['unistd.h', 'fork'],
    'HAVE_CLOCK_GETTIME': ['time.h', 'clock_gettime'],
    'HAVE_ACCEPT4': ['sys/socket.h', 'accept4'],
    'HAVE_STRNLEN': ['string.h', 'strnlen'],
    'HAVE_IF_NAMETOINDEX': ['net/if.h', 'if_nametoindex'],
    'ZMQ_HAVE_STRLCPY': ['string.h', 'strlcpy'],
    'ZMQ_HAVE_SOCK_CLOEXEC': ['sys/socket.h', 'SOCK_CLOEXEC'],
    'ZMQ_HAVE_O_CLOEXEC': ['fcntl.h', 'O_CLOEXEC'],
    'ZMQ_HAVE_SO_KEEPALIVE': ['sys/socket.h', 'SO_KEEPALIVE'],
    'ZMQ_HAVE_SO_BINDTODEVICE': ['sys/socket.h', 'SO_BINDTODEVICE'],
    'ZMQ_HAVE_SO_PRIORITY': ['sys/socket.h', 'SO_PRIORITY'],
    'ZMQ_HAVE_SO_PEERCRED': ['sys/socket.h', 'SO_PEERCRED'],
    'ZMQ_HAVE_BUSY_POLL': ['sys/socket.h', 'SO_BUSY_POLL'],
    'ZMQ_HAVE_TCP_KEEPCNT': ['netinet/tcp.h', 'TCP_KEEPCNT'],
    'ZMQ_HAVE_TCP_KEEPIDLE': ['netinet/tcp.h', 'TCP_KEEPIDLE'],
    'ZMQ_HAVE_TCP_KEEPINTVL': ['netinet/tcp.h', 'TCP_KEEPINTVL'],
    'ZMQ_HAVE_TCP_KEEPALIVE': ['netinet/tcp.h', 'TCP_KEEPALIVE'],
  }
  foreach name, check : checks
    if cxx.has_header_symbol(check[0], check[1], args: gnu)
      conf.set(name, 1)
    endif
  endforeach

  foreach header : ['ifaddrs.h', 'sys/uio.h']
    if cxx.has_header(header)
      conf.set(header == 'ifaddrs.h' ? 'ZMQ_HAVE_IFADDRS' : 'ZMQ_HAVE_UIO', 1)
    endif
  endforeach

  if cxx.has_header_symbol('sys/eventfd.h', 'eventfd')
    conf.set('ZMQ_HAVE_EVENTFD', 1)
    if cxx.has_header_symbol('sys/eventfd.h', 'EFD_CLOEXEC')
      conf.set('ZMQ_HAVE_EVENTFD_CLOEXEC', 1)
    endif
  endif

  thread_dep = dependency('threads')
  setname = {
    'ZMQ_HAVE_PTHREAD_SETNAME_1': 'pthread_setname_np("foo");',
    'ZMQ_HAVE_PTHREAD_SETNAME_2': 'pthread_setname_np(pthread_self(), "foo");',
    'ZMQ_HAVE_PTHREAD_SETNAME_3': 'pthread_setname_np(pthread_self(), "foo", (void *)0);',
    'ZMQ_HAVE_PTHREAD_SET_NAME': 'pthread_set_name_np(pthread_self(), "foo");',
    'ZMQ_HAVE_PTHREAD_SET_AFFINITY': 'cpu_set_t test; pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &test);',
  }
  foreach name, body : setname
    if cxx.compiles('#include <pthread.h>\nint main() { ' + body + ' return 0; }',
                    args: gnu + ['-Werror'], dependencies: thread_dep, name: name)
      conf.set(name, 1)
    endif
  endforeach
endif

platform_hpp = configure_file(
  input: 'builds/cmake/platform.hpp.in',
  output: 'platform.hpp',
  format: 'cmake@',
  configuration: conf,
)

sources = files(
  'src/precompiled.cpp', 'src/address.cpp', 'src/channel.cpp', 'src/client.cpp',
  'src/clock.cpp', 'src/ctx.cpp', 'src/curve_mechanism_base.cpp', 'src/curve_client.cpp',
  'src/curve_server.cpp', 'src/dealer.cpp', 'src/devpoll.cpp', 'src/dgram.cpp',
  'src/dist.cpp', 'src/endpoint.cpp', 'src/epoll.cpp', 'src/err.cpp', 'src/fq.cpp',
  'src/io_object.cpp', 'src/io_thread.cpp', 'src/ip.cpp', 'src/ipc_address.cpp',
  'src/ipc_connecter.cpp', 'src/ipc_listener.cpp', 'src/kqueue.cpp', 'src/lb.cpp',
  'src/mailbox.cpp', 'src/mailbox_safe.cpp', 'src/mechanism.cpp', 'src/mechanism_base.cpp',
  'src/metadata.cpp', 'src/msg.cpp', 'src/mtrie.cpp', 'src/norm_engine.cpp',
  'src/object.cpp', 'src/options.cpp', 'src/own.cpp', 'src/null_mechanism.cpp',
  'src/pair.cpp', 'src/peer.cpp', 'src/pgm_receiver.cpp', 'src/pgm_sender.cpp',
  'src/pgm_socket.cpp', 'src/pipe.cpp', 'src/plain_client.cpp', 'src/plain_server.cpp',
  'src/poll.cpp', 'src/poller_base.cpp', 'src/polling_util.cpp', 'src/pollset.cpp',
  'src/proxy.cpp', 'src/pub.cpp', 'src/pull.cpp', 'src/push.cpp', 'src/random.cpp',
  'src/raw_encoder.cpp', 'src/raw_decoder.cpp', 'src/raw_engine.cpp', 'src/reaper.cpp',
  'src/rep.cpp', 'src/req.cpp', 'src/router.cpp', 'src/select.cpp', 'src/server.cpp',
  'src/session_base.cpp', 'src/signaler.cpp', 'src/socket_base.cpp', 'src/socks.cpp',
  'src/socks_connecter.cpp', 'src/stream.cpp', 'src/stream_engine_base.cpp',
  'src/stream_connecter_base.cpp', 'src/stream_listener_base.cpp', 'src/sub.cpp',
  'src/tcp.cpp', 'src/tcp_address.cpp', 'src/tcp_connecter.cpp', 'src/tcp_listener.cpp',
  'src/thread.cpp', 'src/trie.cpp', 'src/radix_tree.cpp', 'src/v1_decoder.cpp',
  'src/v1_encoder.cpp', 'src/v2_decoder.cpp', 'src/v2_encoder.cpp', 'src/v3_1_encoder.cpp',
  'src/xpub.cpp', 'src/xsub.cpp', 'src/zmq.cpp', 'src/zmq_utils.cpp',
  'src/decoder_allocators.cpp', 'src/socket_poller.cpp', 'src/timers.cpp', 'src/radio.cpp',
  'src/dish.cpp', 'src/udp_engine.cpp', 'src/udp_address.cpp', 'src/scatter.cpp',
  'src/gather.cpp', 'src/ip_resolver.cpp', 'src/zap_client.cpp', 'src/zmtp_engine.cpp',
)

cpp_args = ['-DZMQ_CUSTOM_PLATFORM_HPP', '-DZMQ_STATIC', '-D_REENTRANT', '-D_THREAD_SAFE']
deps = []

if is_windows
  sources += files('external/wepoll/wepoll.c')
  cpp_args += ['-D_WIN32_WINNT=0x0A00', '-DNOMINMAX', '-D_CRT_SECURE_NO_WARNINGS']
  deps += [
    cxx.find_library('ws2_32'),
    cxx.find_library('iphlpapi'),
    cxx.find_library('rpcrt4'),
  ]
else
  if is_linux
    cpp_args += ['-D_GNU_SOURCE']
  endif
  deps += dependency('threads')
endif

warning_args = cxx.get_supported_arguments(['-w'])

libzmq = static_library(
  'zmq',
  sources,
  platform_hpp,
  include_directories: include_directories('include', 'src'),
  cpp_args: cpp_args + warning_args,
  c_args: cpp_args + warning_args,
  dependencies: deps,
  gnu_symbol_visibility: 'hidden',
  pic: true,
)

libzmq_dep = declare_dependency(
  include_directories: include_directories('include'),
  compile_args: ['-DZMQ_STATIC'],
  link_with: libzmq,
  dependencies: deps,
)
