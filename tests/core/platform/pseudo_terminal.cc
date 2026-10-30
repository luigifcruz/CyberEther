#include <catch2/catch_test_macros.hpp>

#include "jetstream/platform.hh"

#include <chrono>
#include <functional>
#include <string>
#include <thread>

using namespace Jetstream;

namespace {

constexpr const char* kTags = "[core][platform][pseudo-terminal]";

bool Drain(Platform::PseudoTerminal& terminal, std::string& output, const std::function<bool()>& done) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (std::chrono::steady_clock::now() < deadline) {
        (void)terminal.read(output, 1024 * 1024);
        if (done()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return false;
}

}  // namespace

TEST_CASE("Pseudo terminals run a program and report when it exits", kTags) {
    if (!Platform::PseudoTerminal::Supported()) {
        SKIP("Pseudo terminals are not supported on this platform.");
    }
    Platform::PseudoTerminal terminal;
    REQUIRE(terminal.start({
        .program = "/bin/sh",
        .arguments = {"-c", "printf 'ready:%s' \"$CYBERETHER_TEST\""},
        .environment = {{"CYBERETHER_TEST", "value"}},
    }) == Result::SUCCESS);

    std::string output;
    CHECK(Drain(terminal, output, [&] { return !terminal.running(); }));
    CHECK(output.find("ready:value") != std::string::npos);
    CHECK_FALSE(terminal.running());
    CHECK(terminal.write("late") == Result::ERROR);
}

TEST_CASE("Pseudo terminals echo input and apply their window size", kTags) {
    if (!Platform::PseudoTerminal::Supported()) {
        SKIP("Pseudo terminals are not supported on this platform.");
    }
    Platform::PseudoTerminal terminal;
    REQUIRE(terminal.start({.program = "/bin/sh", .columns = 80, .rows = 24}) == Result::SUCCESS);
    REQUIRE(terminal.running());

    REQUIRE(terminal.resize(97, 31, 970, 620) == Result::SUCCESS);
    REQUIRE(terminal.write("stty size; printf 'mark%s\\n' done\n") == Result::SUCCESS);

    std::string output;
    CHECK(Drain(terminal, output, [&] { return output.find("markdone") != std::string::npos; }));
    CHECK(output.find("31 97") != std::string::npos);

    terminal.stop();
    CHECK_FALSE(terminal.running());
    CHECK(terminal.resize(80, 24, 0, 0) == Result::ERROR);
}

TEST_CASE("Pseudo terminals block until output arrives or another thread interrupts", kTags) {
    if (!Platform::PseudoTerminal::Supported()) {
        SKIP("Pseudo terminals are not supported on this platform.");
    }
    Platform::PseudoTerminal terminal;
    REQUIRE(terminal.start({
        .program = "/bin/sh",
        .arguments = {"-c", "sleep 0.5; printf 'late'; sleep 30"},
    }) == Result::SUCCESS);

    CHECK(terminal.wait(std::chrono::milliseconds(0)) == Result::SUCCESS);

    std::thread interrupter([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        terminal.interrupt();
    });
    const auto begin = std::chrono::steady_clock::now();
    CHECK(terminal.wait(std::nullopt) == Result::SUCCESS);
    interrupter.join();
    CHECK(std::chrono::steady_clock::now() - begin < std::chrono::seconds(5));

    std::string output;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (output.find("late") == std::string::npos && std::chrono::steady_clock::now() < deadline) {
        REQUIRE(terminal.wait(std::chrono::seconds(1)) == Result::SUCCESS);
        (void)terminal.read(output, 1024 * 1024);
    }
    CHECK(output.find("late") != std::string::npos);

    terminal.stop();
    CHECK(terminal.wait(std::chrono::milliseconds(0)) == Result::ERROR);
    terminal.interrupt();
}

TEST_CASE("Pseudo terminals refuse an empty program", kTags) {
    Platform::PseudoTerminal terminal;
    CHECK(terminal.start({}) == Result::ERROR);
    CHECK_FALSE(terminal.running());
}

TEST_CASE("Pseudo terminals keep unread output readable after the program exits during an input wait", kTags) {
    if (!Platform::PseudoTerminal::Supported()) {
        SKIP("Pseudo terminals are not supported on this platform.");
    }
    Platform::PseudoTerminal terminal;
    REQUIRE(terminal.start({
        .program = "/bin/sh",
        .arguments = {"-c", "stty -echo; sleep 0.3; printf 'tail'; sleep 0.3"},
    }) == Result::SUCCESS);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    REQUIRE(terminal.write(std::string(1024 * 1024, 'x')) == Result::SUCCESS);
    std::this_thread::sleep_for(std::chrono::seconds(1));

    REQUIRE(terminal.wait(std::chrono::milliseconds(100), false) == Result::SUCCESS);
    std::string output;
    CHECK(Drain(terminal, output, [&] { return output.find("tail") != std::string::npos; }));
}
