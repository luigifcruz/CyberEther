#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "jetstream/platform.hh"

using namespace Jetstream;

namespace {

std::string UniqueScheme(const char* label) {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::string("ce-test-") + label + "-" + std::to_string(stamp);
}

struct Received {
    std::mutex mutex;
    std::condition_variable changed;
    std::vector<std::string> urls;

    Platform::UrlScheme::Handler handler() {
        return [this](std::string url) {
            std::lock_guard lock(mutex);
            urls.push_back(std::move(url));
            changed.notify_all();
        };
    }

    bool wait(const U64 count) {
        std::unique_lock lock(mutex);
        return changed.wait_for(lock, std::chrono::seconds(5), [&] { return urls.size() >= count; });
    }
};

}  // namespace

TEST_CASE("URL scheme forwards skip when no instance is listening", "[core][platform][url-scheme]") {
    CHECK(Platform::UrlScheme::Forward(UniqueScheme("idle"), "cyberether://flowgraph/first-steps") == Result::SKIP);
}

TEST_CASE("URL scheme listeners receive forwarded links in order", "[core][platform][url-scheme]") {
    const auto scheme = UniqueScheme("forward");
    Received received;
    Platform::UrlScheme listener;
    REQUIRE(listener.listen(scheme, received.handler()) == Result::SUCCESS);

    REQUIRE(Platform::UrlScheme::Forward(scheme, "cyberether://flowgraph/first-steps") == Result::SUCCESS);
    REQUIRE(received.wait(1));
    REQUIRE(Platform::UrlScheme::Forward(scheme, "cyberether://flowgraph/python-qpsk") == Result::SUCCESS);
    REQUIRE(received.wait(2));
    CHECK(received.urls == std::vector<std::string>{"cyberether://flowgraph/first-steps",
                                                    "cyberether://flowgraph/python-qpsk"});

    CHECK(Platform::UrlScheme::Forward(scheme, std::string(4096, 'a')) == Result::ERROR);

    listener.close();
    CHECK(Platform::UrlScheme::Forward(scheme, "cyberether://flowgraph/first-steps") == Result::SKIP);
}

TEST_CASE("URL scheme endpoints have one owner and can be taken over after it closes",
          "[core][platform][url-scheme]") {
    const auto scheme = UniqueScheme("owner");
    Received first;
    Received second;
    Platform::UrlScheme owner;
    Platform::UrlScheme other;
    REQUIRE(owner.listen(scheme, first.handler()) == Result::SUCCESS);
    CHECK(other.listen(scheme, second.handler()) == Result::SKIP);

    REQUIRE(Platform::UrlScheme::Forward(scheme, "cyberether://flowgraph/first-steps") == Result::SUCCESS);
    REQUIRE(first.wait(1));
    CHECK(second.urls.empty());

    owner.close();
    REQUIRE(other.listen(scheme, second.handler()) == Result::SUCCESS);
    REQUIRE(Platform::UrlScheme::Forward(scheme, "cyberether://flowgraph/python-qpsk") == Result::SUCCESS);
    REQUIRE(second.wait(1));
    CHECK(second.urls.front() == "cyberether://flowgraph/python-qpsk");
}
