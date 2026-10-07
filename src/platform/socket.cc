#include "jetstream/platform.hh"

#if defined(JST_OS_WINDOWS)
#include <winsock2.h>
#else
#include <csignal>
#include <sys/socket.h>
#endif

namespace Jetstream::Platform {

bool ShutdownSocketRead(const std::uintptr_t socket) noexcept {
#if defined(JST_OS_WINDOWS)
    return shutdown(static_cast<SOCKET>(socket), SD_RECEIVE) == 0;
#else
    return shutdown(static_cast<int>(socket), SHUT_RD) == 0;
#endif
}

void DisableSocketSigPipe([[maybe_unused]] const std::uintptr_t socket) noexcept {
#if defined(SO_NOSIGPIPE)
    const int enabled = 1;
    setsockopt(static_cast<int>(socket), SOL_SOCKET, SO_NOSIGPIPE, &enabled, sizeof(enabled));
#endif
}

void IgnoreBrokenPipe() noexcept {
#if !defined(JST_OS_WINDOWS)
    std::signal(SIGPIPE, SIG_IGN);
#endif
}

}  // namespace Jetstream::Platform
