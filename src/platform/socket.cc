#include "jetstream/platform.hh"

#if defined(JST_OS_WINDOWS)
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#undef ERROR
#undef FATAL
#else
#include <arpa/inet.h>
#include <csignal>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
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

Result LocalNetworkAddress(std::string& address) {
#if defined(JST_OS_WINDOWS)
    WSADATA data;
    if (WSAStartup(MAKEWORD(2, 2), &data) != 0) {
        return Result::ERROR;
    }
    const SOCKET handle = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    const bool valid = handle != INVALID_SOCKET;
#else
    const int handle = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    const bool valid = handle >= 0;
#endif

    bool found = false;
    if (valid) {
        sockaddr_in remote{};
        remote.sin_family = AF_INET;
        remote.sin_port = htons(53);
        inet_pton(AF_INET, "192.0.2.1", &remote.sin_addr);

        sockaddr_in local{};
#if defined(JST_OS_WINDOWS)
        int length = sizeof(local);
#else
        socklen_t length = sizeof(local);
#endif
        char text[INET_ADDRSTRLEN] = {};
        found = connect(handle, reinterpret_cast<const sockaddr*>(&remote), sizeof(remote)) == 0 &&
                getsockname(handle, reinterpret_cast<sockaddr*>(&local), &length) == 0 &&
                local.sin_addr.s_addr != htonl(INADDR_ANY) &&
                inet_ntop(AF_INET, &local.sin_addr, text, sizeof(text)) != nullptr;
        if (found) {
            address = text;
        }
    }

#if defined(JST_OS_WINDOWS)
    if (valid) {
        closesocket(handle);
    }
    WSACleanup();
#else
    if (valid) {
        close(handle);
    }
#endif

    return found ? Result::SUCCESS : Result::ERROR;
}

}  // namespace Jetstream::Platform
