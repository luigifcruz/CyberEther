#include <catch2/catch_test_macros.hpp>

#include "jetstream/platform.hh"

#include <chrono>
#include <functional>
#include <string>
#include <thread>

#if defined(JST_OS_LINUX) || defined(JST_OS_MAC)
#include <cerrno>
#include <csignal>
#include <sys/wait.h>
#endif

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
        (void)terminal.wait(std::chrono::milliseconds(10));
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
        .arguments = {"-c", "stty -echo; printf 'ready'; IFS= read -r step; printf 'late'; IFS= read -r step"},
    }) == Result::SUCCESS);

    std::string output;
    REQUIRE(Drain(terminal, output, [&] { return output.find("ready") != std::string::npos; }));
    output.clear();
    CHECK(terminal.wait(std::chrono::milliseconds(0)) == Result::SUCCESS);

    // Interrupts must wake the wait even when queued before it enters poll().
    std::thread interrupter([&] { terminal.interrupt(); });
    CHECK(terminal.wait(std::nullopt) == Result::SUCCESS);
    interrupter.join();
    REQUIRE(terminal.read(output, 1024) == Result::SUCCESS);
    CHECK(output.empty());

    REQUIRE(terminal.write("\n") == Result::SUCCESS);
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

TEST_CASE("Pseudo terminals keep the program's final output readable during an input wait", kTags) {
#if defined(JST_OS_LINUX) || defined(JST_OS_MAC)
    if (!Platform::PseudoTerminal::Supported()) {
        SKIP("Pseudo terminals are not supported on this platform.");
    }
    Platform::PseudoTerminal terminal;
    REQUIRE(terminal.start({
        .program = "/bin/sh",
        .arguments = {"-c", "stty raw -echo; printf '%s\\n' \"$$\"; kill -STOP \"$$\"; "
                            "printf 'tail'; kill -STOP \"$$\""},
    }) == Result::SUCCESS);
    std::string output;
    REQUIRE(Drain(terminal, output, [&] { return output.find('\n') != std::string::npos; }));
    const auto process = static_cast<pid_t>(std::stol(output));
    output.clear();
    const auto waitFor = [&](int event) {
        siginfo_t info{};
        int result;
        do {
            result = waitid(P_PID, process, &info, event);
        } while (result < 0 && errno == EINTR);
        REQUIRE(result == 0);
        REQUIRE(info.si_pid == process);
        return info.si_code;
    };

    // Hold the child without consuming input so the write fills the PTY and
    // leaves pending bytes. A second stop confirms the final output is queued.
    REQUIRE(waitFor(WSTOPPED) == CLD_STOPPED);
    REQUIRE(terminal.write(std::string(1024 * 1024, 'x')) == Result::SUCCESS);
    REQUIRE(kill(process, SIGCONT) == 0);
    REQUIRE(waitFor(WSTOPPED) == CLD_STOPPED);
    REQUIRE(kill(process, SIGCONT) == 0);
#if defined(JST_OS_LINUX)
    // Leave reaping to the terminal. Darwin's last slave close can wait for
    // output to drain, so waiting for exit before reading would deadlock there.
    REQUIRE(waitFor(WEXITED | WNOWAIT) == CLD_EXITED);
#endif

    REQUIRE(terminal.wait(std::chrono::milliseconds(0), false) == Result::SUCCESS);
    CHECK(Drain(terminal, output, [&] { return output.find("tail") != std::string::npos; }));
#else
    SKIP("Pseudo terminals are not supported on this platform.");
#endif
}
