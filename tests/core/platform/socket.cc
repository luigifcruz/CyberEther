#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <array>
#include <vector>

#include "jetstream/platform.hh"

#if defined(JST_OS_WINDOWS)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#undef ERROR
#undef FATAL
#else
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#endif

using namespace Jetstream;

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

struct NativeSocket {
    SocketHandle handle;

    explicit NativeSocket(const int family) : handle(::socket(family, SOCK_DGRAM, IPPROTO_UDP)) {
        REQUIRE(handle != InvalidSocket);
    }

    ~NativeSocket() {
#if defined(JST_OS_WINDOWS)
        closesocket(handle);
#else
        ::close(handle);
#endif
    }

    NativeSocket(const NativeSocket&) = delete;
    NativeSocket& operator=(const NativeSocket&) = delete;
};

struct UdpPeer {
    Platform::UdpSocket sender;
    NativeSocket receiver;
    Platform::UdpSocket::Config config;

    explicit UdpPeer(const int family) : receiver(family) {
        sockaddr_storage address{};
        SocketLength length;
        if (family == AF_INET) {
            auto& ipv4 = reinterpret_cast<sockaddr_in&>(address);
            ipv4.sin_family = AF_INET;
            ipv4.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
            length = sizeof(ipv4);
            config.address = "127.0.0.1";
        } else {
            auto& ipv6 = reinterpret_cast<sockaddr_in6&>(address);
            ipv6.sin6_family = AF_INET6;
            ipv6.sin6_addr = in6addr_loopback;
            length = sizeof(ipv6);
            config.address = "::1";
        }
        REQUIRE(::bind(receiver.handle, reinterpret_cast<const sockaddr*>(&address), length) == 0);
        REQUIRE(getsockname(receiver.handle, reinterpret_cast<sockaddr*>(&address), &length) == 0);
        config.port = family == AF_INET ? ntohs(reinterpret_cast<const sockaddr_in&>(address).sin_port)
                                       : ntohs(reinterpret_cast<const sockaddr_in6&>(address).sin6_port);
        config.timeoutMilliseconds = 100;
#if defined(JST_OS_WINDOWS)
        const DWORD timeout = 1000;
#else
        const timeval timeout{1, 0};
#endif
        REQUIRE(setsockopt(receiver.handle, SOL_SOCKET, SO_RCVTIMEO,
                          reinterpret_cast<const char*>(&timeout), sizeof(timeout)) == 0);
    }
};

}

TEST_CASE("Connected UDP sockets preserve datagrams and can change destinations",
          "[core][platform][udp]") {
    const auto family = GENERATE(AF_INET, AF_INET6);
    UdpPeer first(family);
    UdpPeer second(family);
    std::string error;
    for (const auto* peer : {&first, &second}) {
        REQUIRE(first.sender.connect(peer->config, error) == Result::SUCCESS);
        REQUIRE(first.sender.open());
        for (const int size : {0, 1, 1472}) {
            const std::array<char, 1472> payload{'u', 'd', 'p'};
            REQUIRE(first.sender.send(payload.data(), size) == Result::SUCCESS);
            std::array<char, 1472> received{};
            REQUIRE(::recvfrom(peer->receiver.handle, received.data(), static_cast<int>(received.size()), 0,
                               nullptr, nullptr) == size);
            REQUIRE(std::equal(payload.begin(), payload.begin() + size, received.begin()));
        }
    }

    const std::vector<char> oversized(65536);
    REQUIRE(first.sender.send(oversized.data(), oversized.size()) == Result::ERROR);
    const char payload = 'u';
    REQUIRE(first.sender.send(&payload, 1) == Result::SUCCESS);
    char received = 0;
    REQUIRE(::recvfrom(second.receiver.handle, &received, 1, 0, nullptr, nullptr) == 1);
    REQUIRE(received == payload);

    first.sender.close();
    REQUIRE_FALSE(first.sender.open());
    REQUIRE(first.sender.send(&payload, 1) == Result::ERROR);
}

TEST_CASE("Connected UDP sockets accept replies only from their peer",
          "[core][platform][udp]") {
    const auto family = GENERATE(AF_INET, AF_INET6);
    UdpPeer peer(family);
    NativeSocket other(family);
    std::string error;
    REQUIRE(peer.sender.connect(peer.config, error) == Result::SUCCESS);
    const char payload = 'u';
    REQUIRE(peer.sender.send(&payload, 1) == Result::SUCCESS);

    sockaddr_storage source{};
    SocketLength length = sizeof(source);
    char received = 0;
    REQUIRE(::recvfrom(peer.receiver.handle, &received, 1, 0,
                       reinterpret_cast<sockaddr*>(&source), &length) == 1);
    const char unexpected = 'x';
    REQUIRE(::sendto(other.handle, &unexpected, 1, 0,
                     reinterpret_cast<const sockaddr*>(&source), length) == 1);
    REQUIRE(::sendto(peer.receiver.handle, &payload, 1, 0,
                     reinterpret_cast<const sockaddr*>(&source), length) == 1);
    U64 size = 0;
    REQUIRE(peer.sender.receive(&received, 1, size) == Result::SUCCESS);
    REQUIRE(size == 1);
    REQUIRE(received == payload);
}

TEST_CASE("UDP binding and failed setup clear the connected state",
          "[core][platform][udp]") {
    UdpPeer peer(AF_INET);
    std::string error;
    REQUIRE(peer.sender.connect(peer.config, error) == Result::SUCCESS);

    SECTION("binding replaces the connected socket") {
        REQUIRE(peer.sender.bind({.address = "127.0.0.1"}, error) == Result::SUCCESS);
        REQUIRE(peer.sender.open());
    }
    SECTION("failed connection setup closes the socket") {
        auto config = peer.config;
        config.interfaceName = "cyberether-missing-interface";
        REQUIRE(peer.sender.connect(config, error) == Result::ERROR);
        REQUIRE_FALSE(peer.sender.open());
        REQUIRE_FALSE(error.empty());
    }

    const char payload = 'u';
    REQUIRE(peer.sender.send(&payload, 1) == Result::ERROR);
}
