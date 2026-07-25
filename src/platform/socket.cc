#include "jetstream/platform.hh"

#if defined(JST_OS_WINDOWS)
#include <winsock2.h>
#else
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

}  // namespace Jetstream::Platform
