#include "jetstream/platform.hh"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <iterator>
#include <limits>
#include <system_error>

#if defined(JST_OS_WINDOWS)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <iphlpapi.h>
#undef ERROR
#undef FATAL
#else
#include <arpa/inet.h>
#include <csignal>
#include <ifaddrs.h>
#include <net/if.h>
#include <netdb.h>
#include <netinet/in.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#endif

namespace Jetstream::Platform {

namespace {

#if defined(JST_OS_WINDOWS)
using SocketHandle = SOCKET;
using SocketLength = int;
constexpr SocketHandle InvalidSocket = INVALID_SOCKET;
#else
using SocketHandle = int;
using SocketLength = socklen_t;
constexpr SocketHandle InvalidSocket = -1;
#endif

struct InterfaceRecord {
    std::string name;
    U32 index = 0;
    std::vector<sockaddr_storage> addresses;
};

int LastSocketError() {
#if defined(JST_OS_WINDOWS)
    return WSAGetLastError();
#else
    return errno;
#endif
}

bool SocketTimedOut(const int code) {
#if defined(JST_OS_WINDOWS)
    return code == WSAETIMEDOUT || code == WSAEWOULDBLOCK;
#else
    return code == EAGAIN || code == EWOULDBLOCK || code == EINTR;
#endif
}

template<typename T>
bool SetSocketOption(const SocketHandle handle, const int level, const int name, const T& value) {
    return setsockopt(handle, level, name, reinterpret_cast<const char*>(&value), sizeof(value)) == 0;
}

bool SetSocketTimeout(const SocketHandle handle, const int name, const U64 milliseconds) {
#if defined(JST_OS_WINDOWS)
    const DWORD timeout = static_cast<DWORD>(std::min<U64>(milliseconds, std::numeric_limits<DWORD>::max()));
#else
    timeval timeout{};
    timeout.tv_sec = static_cast<decltype(timeout.tv_sec)>(milliseconds / 1000);
    timeout.tv_usec = static_cast<decltype(timeout.tv_usec)>((milliseconds % 1000) * 1000);
#endif
    return SetSocketOption(handle, SOL_SOCKET, name, timeout);
}

void SetSocketBuffer(const SocketHandle handle, const int name, const U64 bytes) {
    if (bytes == 0) {
        return;
    }
    int size = static_cast<int>(std::min<U64>(bytes, std::numeric_limits<int>::max()));
    while (size >= 64 * 1024 && !SetSocketOption(handle, SOL_SOCKET, name, size)) {
        size /= 2;
    }
}

const in_addr& Ipv4(const sockaddr_storage& address) {
    return reinterpret_cast<const sockaddr_in&>(address).sin_addr;
}

const in6_addr& Ipv6(const sockaddr_storage& address) {
    return reinterpret_cast<const sockaddr_in6&>(address).sin6_addr;
}

bool IsMulticast(const sockaddr_storage& address) {
    if (address.ss_family == AF_INET) {
        return (ntohl(Ipv4(address).s_addr) & 0xF0000000u) == 0xE0000000u;
    }
    if (address.ss_family == AF_INET6) {
        return Ipv6(address).s6_addr[0] == 0xFF;
    }
    return false;
}

bool IsLinkLocal(const sockaddr_storage& address) {
    const auto& bytes = Ipv6(address).s6_addr;
    return address.ss_family == AF_INET6 && bytes[0] == 0xFE && (bytes[1] & 0xC0) == 0x80;
}

sockaddr_storage Unmapped(const sockaddr_storage& address) {
    const auto& bytes = Ipv6(address).s6_addr;
    constexpr U8 prefix[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xFF, 0xFF};
    if (address.ss_family != AF_INET6 || std::memcmp(bytes, prefix, sizeof(prefix)) != 0) {
        return address;
    }

    sockaddr_storage ipv4{};
    auto& target = reinterpret_cast<sockaddr_in&>(ipv4);
    target.sin_family = AF_INET;
    std::memcpy(&target.sin_addr, bytes + 12, sizeof(target.sin_addr));
    return ipv4;
}

bool SameHost(const sockaddr_storage& a, const sockaddr_storage& b) {
    const sockaddr_storage left = Unmapped(a);
    const sockaddr_storage right = Unmapped(b);
    if (left.ss_family != right.ss_family) {
        return false;
    }
    if (left.ss_family == AF_INET) {
        return Ipv4(left).s_addr == Ipv4(right).s_addr;
    }
    if (left.ss_family != AF_INET6 || std::memcmp(&Ipv6(left), &Ipv6(right), sizeof(in6_addr)) != 0) {
        return false;
    }
    const auto scope = reinterpret_cast<const sockaddr_in6&>(right).sin6_scope_id;
    return scope == 0 || reinterpret_cast<const sockaddr_in6&>(left).sin6_scope_id == scope;
}

std::string FormatAddress(const sockaddr_storage& address) {
    char text[NI_MAXHOST] = {};
    const SocketLength length = address.ss_family == AF_INET ? sizeof(sockaddr_in) : sizeof(sockaddr_in6);
    if (getnameinfo(reinterpret_cast<const sockaddr*>(&address), length,
                    text, sizeof(text), nullptr, 0, NI_NUMERICHOST) != 0) {
        return {};
    }
    return text;
}

Result ResolveAddress(const std::string& address,
                      const U16 port,
                      const bool passive,
                      const int family,
                      sockaddr_storage& resolved,
                      SocketLength& length) {
    addrinfo hints{};
    hints.ai_family = family;
    hints.ai_socktype = SOCK_DGRAM;
    hints.ai_protocol = IPPROTO_UDP;
    hints.ai_flags = passive ? AI_PASSIVE : 0;

    addrinfo* results = nullptr;
    const std::string service = std::to_string(port);
    const char* node = address.empty() ? nullptr : address.c_str();
    if (getaddrinfo(node, service.c_str(), &hints, &results) != 0 || !results) {
        return Result::ERROR;
    }

    std::memcpy(&resolved, results->ai_addr, results->ai_addrlen);
    length = static_cast<SocketLength>(results->ai_addrlen);
    freeaddrinfo(results);

    return Result::SUCCESS;
}

Result ResolveHosts(const std::string& address, const int family, std::vector<sockaddr_storage>& resolved) {
    addrinfo hints{};
    hints.ai_family = family;
    hints.ai_socktype = SOCK_DGRAM;
    hints.ai_protocol = IPPROTO_UDP;

    addrinfo* results = nullptr;
    if (getaddrinfo(address.c_str(), nullptr, &hints, &results) != 0 || !results) {
        return Result::ERROR;
    }

    resolved.clear();
    for (const auto* entry = results; entry; entry = entry->ai_next) {
        sockaddr_storage host{};
        std::memcpy(&host, entry->ai_addr, std::min<std::size_t>(entry->ai_addrlen, sizeof(host)));
        resolved.push_back(host);
    }
    freeaddrinfo(results);

    return Result::SUCCESS;
}

std::vector<InterfaceRecord> CollectInterfaces() {
    std::vector<InterfaceRecord> records;

#if defined(JST_OS_WINDOWS)
    ULONG size = 16 * 1024;
    std::vector<U8> buffer;
    ULONG status = ERROR_BUFFER_OVERFLOW;
    for (int attempt = 0; attempt < 3 && status == ERROR_BUFFER_OVERFLOW; ++attempt) {
        buffer.resize(size);
        status = GetAdaptersAddresses(AF_UNSPEC,
                                      GAA_FLAG_SKIP_ANYCAST | GAA_FLAG_SKIP_MULTICAST | GAA_FLAG_SKIP_DNS_SERVER,
                                      nullptr,
                                      reinterpret_cast<IP_ADAPTER_ADDRESSES*>(buffer.data()),
                                      &size);
    }
    if (status != NO_ERROR) {
        return records;
    }

    for (auto* adapter = reinterpret_cast<IP_ADAPTER_ADDRESSES*>(buffer.data()); adapter; adapter = adapter->Next) {
        if (adapter->OperStatus != IfOperStatusUp) {
            continue;
        }

        InterfaceRecord record;
        record.name = PathToUtf8(std::filesystem::path(adapter->FriendlyName));
        record.index = adapter->Ipv6IfIndex != 0 ? adapter->Ipv6IfIndex : adapter->IfIndex;
        for (auto* unicast = adapter->FirstUnicastAddress; unicast; unicast = unicast->Next) {
            const auto* address = unicast->Address.lpSockaddr;
            if (!address || (address->sa_family != AF_INET && address->sa_family != AF_INET6)) {
                continue;
            }
            sockaddr_storage entry{};
            std::memcpy(&entry, address, std::min<std::size_t>(unicast->Address.iSockaddrLength, sizeof(entry)));
            record.addresses.push_back(entry);
        }
        records.push_back(std::move(record));
    }
#else
    ifaddrs* list = nullptr;
    if (getifaddrs(&list) != 0) {
        return records;
    }

    for (auto* entry = list; entry; entry = entry->ifa_next) {
        if (!entry->ifa_addr || !(entry->ifa_flags & IFF_UP)) {
            continue;
        }
        const int family = entry->ifa_addr->sa_family;
        if (family != AF_INET && family != AF_INET6) {
            continue;
        }

        auto record = std::find_if(records.begin(), records.end(), [&](const auto& candidate) {
            return candidate.name == entry->ifa_name;
        });
        if (record == records.end()) {
            records.push_back({entry->ifa_name, if_nametoindex(entry->ifa_name), {}});
            record = std::prev(records.end());
        }

        sockaddr_storage address{};
        std::memcpy(&address, entry->ifa_addr, family == AF_INET ? sizeof(sockaddr_in) : sizeof(sockaddr_in6));
        record->addresses.push_back(address);
    }

    freeifaddrs(list);
#endif

    return records;
}

Result FindInterface(const std::string& name, InterfaceRecord& record, std::string& error) {
    for (auto& candidate : CollectInterfaces()) {
        if (candidate.name == name) {
            record = std::move(candidate);
            return Result::SUCCESS;
        }
    }
    error = "Interface not found";
    return Result::ERROR;
}

bool InterfaceAddress(const InterfaceRecord& record, const int family, sockaddr_storage& address) {
    const sockaddr_storage* fallback = nullptr;
    for (const auto& candidate : record.addresses) {
        if (candidate.ss_family != family) {
            continue;
        }
        if (!IsLinkLocal(candidate)) {
            address = candidate;
            return true;
        }
        fallback = fallback ? fallback : &candidate;
    }
    if (fallback) {
        address = *fallback;
    }
    return fallback != nullptr;
}

}  // namespace

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

struct UdpSocket::Impl {
    SocketHandle handle = InvalidSocket;
    sockaddr_storage destination{};
    SocketLength destinationLength = 0;
    std::vector<sockaddr_storage> sources;
    bool filtered = false;

#if defined(JST_OS_WINDOWS)
    bool started = false;

    Impl() {
        WSADATA data;
        started = WSAStartup(MAKEWORD(2, 2), &data) == 0;
    }

    ~Impl() {
        if (started) {
            WSACleanup();
        }
    }
#endif

    Result create(const int family, std::string& error) {
#if defined(SOCK_CLOEXEC)
        handle = socket(family, SOCK_DGRAM | SOCK_CLOEXEC, IPPROTO_UDP);
#else
        handle = socket(family, SOCK_DGRAM, IPPROTO_UDP);
#endif
        if (handle == InvalidSocket) {
            return fail(error);
        }
#if defined(JST_OS_WINDOWS)
        if (!SetHandleInformation(reinterpret_cast<HANDLE>(handle), HANDLE_FLAG_INHERIT, 0)) {
            return fail(error, "Could not disable socket inheritance");
        }
#elif !defined(SOCK_CLOEXEC)
        if (fcntl(handle, F_SETFD, FD_CLOEXEC) != 0) {
            return fail(error);
        }
#endif
        return Result::SUCCESS;
    }

    Result fail(std::string& error) {
        error = std::system_category().message(LastSocketError());
        release();
        return Result::ERROR;
    }

    Result fail(std::string& error, const char* reason) {
        error = reason;
        release();
        return Result::ERROR;
    }

    void release() {
        if (handle != InvalidSocket) {
#if defined(JST_OS_WINDOWS)
            closesocket(handle);
#else
            ::close(handle);
#endif
        }
        handle = InvalidSocket;
        destinationLength = 0;
        filtered = false;
    }
};

Result UdpSocket::ListInterfaces(std::vector<Interface>& interfaces) {
    interfaces.clear();

#if defined(JST_OS_WINDOWS)
    WSADATA data;
    if (WSAStartup(MAKEWORD(2, 2), &data) != 0) {
        return Result::ERROR;
    }
#endif

    for (const auto& record : CollectInterfaces()) {
        Interface entry{.name = record.name};
        for (const bool linkLocal : {false, true}) {
            for (const int family : {AF_INET, AF_INET6}) {
                for (const auto& address : record.addresses) {
                    if (address.ss_family == family && IsLinkLocal(address) == linkLocal) {
                        entry.addresses.push_back(FormatAddress(address));
                    }
                }
            }
        }
        if (!entry.addresses.empty()) {
            interfaces.push_back(std::move(entry));
        }
    }

#if defined(JST_OS_WINDOWS)
    WSACleanup();
#endif

    return Result::SUCCESS;
}

UdpSocket::UdpSocket() : impl(std::make_unique<Impl>()) {}

UdpSocket::~UdpSocket() {
    close();
}

Result UdpSocket::bind(const Config& config, std::string& error) {
    close();

    sockaddr_storage address{};
    SocketLength length = 0;
    if (ResolveAddress(config.address, config.port, true, AF_UNSPEC, address, length) != Result::SUCCESS) {
        error = "Could not resolve the address";
        return Result::ERROR;
    }

    InterfaceRecord selected;
    const bool pinned = !config.interfaceName.empty();
    if (pinned) {
        JST_CHECK(FindInterface(config.interfaceName, selected, error));
    }

    const bool multicast = IsMulticast(address);
    sockaddr_storage local = address;
    sockaddr_storage interfaceAddress{};
    const bool hasInterfaceAddress = pinned && InterfaceAddress(selected, address.ss_family, interfaceAddress);
    if (multicast) {
#if defined(JST_OS_WINDOWS)
        if (local.ss_family == AF_INET) {
            reinterpret_cast<sockaddr_in&>(local).sin_addr.s_addr = htonl(INADDR_ANY);
        } else {
            reinterpret_cast<sockaddr_in6&>(local).sin6_addr = in6addr_any;
        }
#else
        if (local.ss_family == AF_INET6 && pinned) {
            reinterpret_cast<sockaddr_in6&>(local).sin6_scope_id = selected.index;
        }
#endif
    }

    if (!config.source.empty()) {
        const int sourceFamily = address.ss_family == AF_INET ? AF_INET : AF_UNSPEC;
        if (ResolveHosts(config.source, sourceFamily, impl->sources) != Result::SUCCESS) {
            error = "Could not resolve the source address";
            return Result::ERROR;
        }
    }

    JST_CHECK(impl->create(address.ss_family, error));
    impl->filtered = !config.source.empty();

    const int enabled = 1;
    if (multicast && !SetSocketOption(impl->handle, SOL_SOCKET, SO_REUSEADDR, enabled)) {
        return impl->fail(error);
    }
#if defined(SO_REUSEPORT)
    if (multicast && !SetSocketOption(impl->handle, SOL_SOCKET, SO_REUSEPORT, enabled)) {
        return impl->fail(error);
    }
#endif
    if (address.ss_family == AF_INET6) {
        const int disabled = 0;
        SetSocketOption(impl->handle, IPPROTO_IPV6, IPV6_V6ONLY, disabled);
    }

    SetSocketBuffer(impl->handle, SO_RCVBUF, config.bufferBytes);
    if (config.timeoutMilliseconds > 0 &&
        !SetSocketTimeout(impl->handle, SO_RCVTIMEO, config.timeoutMilliseconds)) {
        return impl->fail(error);
    }

    if (::bind(impl->handle, reinterpret_cast<const sockaddr*>(&local), length) != 0) {
        return impl->fail(error);
    }

    if (multicast && address.ss_family == AF_INET) {
        if (pinned && !hasInterfaceAddress) {
            return impl->fail(error, "Interface has no IPv4 address");
        }
        ip_mreq request{};
        request.imr_multiaddr = Ipv4(address);
        request.imr_interface.s_addr = pinned ? Ipv4(interfaceAddress).s_addr : htonl(INADDR_ANY);
        if (!SetSocketOption(impl->handle, IPPROTO_IP, IP_ADD_MEMBERSHIP, request)) {
            return impl->fail(error);
        }
    } else if (multicast) {
        ipv6_mreq request{};
        request.ipv6mr_multiaddr = Ipv6(address);
        request.ipv6mr_interface = pinned ? selected.index : reinterpret_cast<const sockaddr_in6&>(address).sin6_scope_id;
        if (!SetSocketOption(impl->handle, IPPROTO_IPV6, IPV6_JOIN_GROUP, request)) {
            return impl->fail(error);
        }
    }

    return Result::SUCCESS;
}

Result UdpSocket::connect(const Config& config, std::string& error) {
    close();

    sockaddr_storage address{};
    SocketLength length = 0;
    if (ResolveAddress(config.address, config.port, false, AF_UNSPEC, address, length) != Result::SUCCESS) {
        error = "Could not resolve the address";
        return Result::ERROR;
    }

    InterfaceRecord selected;
    const bool pinned = !config.interfaceName.empty();
    if (pinned) {
        JST_CHECK(FindInterface(config.interfaceName, selected, error));
    }

    JST_CHECK(impl->create(address.ss_family, error));

    const int enabled = 1;
    if (address.ss_family == AF_INET &&
        !SetSocketOption(impl->handle, SOL_SOCKET, SO_BROADCAST, enabled)) {
        return impl->fail(error);
    }

    if (pinned && IsMulticast(address) && address.ss_family == AF_INET) {
        sockaddr_storage interfaceAddress{};
        if (!InterfaceAddress(selected, AF_INET, interfaceAddress)) {
            return impl->fail(error, "Interface has no IPv4 address");
        }
        if (!SetSocketOption(impl->handle, IPPROTO_IP, IP_MULTICAST_IF, Ipv4(interfaceAddress))) {
            return impl->fail(error);
        }
    } else if (pinned && IsMulticast(address)) {
        const unsigned int index = selected.index;
        if (!SetSocketOption(impl->handle, IPPROTO_IPV6, IPV6_MULTICAST_IF, index)) {
            return impl->fail(error);
        }
    }

    SetSocketBuffer(impl->handle, SO_SNDBUF, config.bufferBytes);
    if (config.timeoutMilliseconds > 0 &&
        !SetSocketTimeout(impl->handle, SO_SNDTIMEO, config.timeoutMilliseconds)) {
        return impl->fail(error);
    }

    impl->destination = address;
    impl->destinationLength = length;

    return Result::SUCCESS;
}

Result UdpSocket::receive(void* data, const U64 size, U64& received) {
    received = 0;
    if (!open()) {
        return Result::ERROR;
    }

    sockaddr_storage sender{};
    SocketLength senderLength = sizeof(sender);
    auto* senderAddress = reinterpret_cast<sockaddr*>(&sender);
#if defined(JST_OS_WINDOWS)
    const int capacity = static_cast<int>(std::min<U64>(size, std::numeric_limits<int>::max()));
    const int count = ::recvfrom(impl->handle, static_cast<char*>(data), capacity, 0, senderAddress, &senderLength);
#else
    const ssize_t count = ::recvfrom(impl->handle, data, size, 0, senderAddress, &senderLength);
#endif
    if (count < 0) {
        return SocketTimedOut(LastSocketError()) ? Result::TIMEOUT : Result::ERROR;
    }

    if (impl->filtered && std::none_of(impl->sources.begin(), impl->sources.end(), [&](const auto& source) {
            return SameHost(sender, source);
        })) {
        return Result::SKIP;
    }

    received = static_cast<U64>(count);
    return Result::SUCCESS;
}

Result UdpSocket::send(const void* data, const U64 size) {
    if (!open() || impl->destinationLength == 0) {
        return Result::ERROR;
    }

    const auto* destination = reinterpret_cast<const sockaddr*>(&impl->destination);
    while (true) {
#if defined(JST_OS_WINDOWS)
        const int count = ::sendto(impl->handle, static_cast<const char*>(data),
                                   static_cast<int>(std::min<U64>(size, std::numeric_limits<int>::max())),
                                   0, destination, impl->destinationLength);
#else
        const ssize_t count = ::sendto(impl->handle, data, size, 0, destination, impl->destinationLength);
#endif
        if (count >= 0) {
            return static_cast<U64>(count) == size ? Result::SUCCESS : Result::ERROR;
        }

        const int code = LastSocketError();
#if !defined(JST_OS_WINDOWS)
        if (code == EINTR) {
            continue;
        }
#endif
        return SocketTimedOut(code) ? Result::TIMEOUT : Result::ERROR;
    }
}

void UdpSocket::close() {
    impl->release();
}

bool UdpSocket::open() const {
    return impl->handle != InvalidSocket;
}

}  // namespace Jetstream::Platform
