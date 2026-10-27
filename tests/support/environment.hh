#ifndef JETSTREAM_TESTS_SUPPORT_ENVIRONMENT_HH
#define JETSTREAM_TESTS_SUPPORT_ENVIRONMENT_HH

#include <catch2/catch_test_macros.hpp>

#include "jetstream/platform.hh"
#include "jetstream/settings.hh"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <optional>
#include <string>
#include <system_error>
#include <utility>

namespace TestSupport {

class ScopedEnvironment {
 public:
    explicit ScopedEnvironment(std::string name) : variable(std::move(name)) {
        std::string value;
        if (Jetstream::Platform::EnvironmentVariable(variable, value) == Jetstream::Result::SUCCESS) {
            previous = std::move(value);
        }
    }

    ~ScopedEnvironment() {
        (void)Jetstream::Platform::WriteEnvironmentVariable(variable, previous);
    }

    ScopedEnvironment(const ScopedEnvironment&) = delete;
    ScopedEnvironment& operator=(const ScopedEnvironment&) = delete;

    bool set(const std::optional<std::string>& value) const {
        return Jetstream::Platform::WriteEnvironmentVariable(variable, value) == Jetstream::Result::SUCCESS;
    }

    const std::string& name() const {
        return variable;
    }

 private:
    std::string variable;
    std::optional<std::string> previous;
};

class SettingsSandbox {
 public:
#if defined(JST_OS_LINUX) || defined(JST_OS_WINDOWS) || defined(JST_OS_MAC)
    static constexpr bool Supported = true;
#else
    static constexpr bool Supported = false;
#endif

    explicit SettingsSandbox(const std::string& label) : directory(Directory(label)) {
#if defined(JST_OS_LINUX)
        redirected = home.set(Jetstream::Platform::PathToUtf8(directory / "home")) &&
                     config.set(Jetstream::Platform::PathToUtf8(directory / "config"));
#elif defined(JST_OS_WINDOWS)
        redirected = appData.set(Jetstream::Platform::PathToUtf8(directory / "AppData" / "Roaming"));
#elif defined(JST_OS_MAC)
        redirected = home.set(Jetstream::Platform::PathToUtf8(directory));
#endif
    }

    ~SettingsSandbox() {
        (void)Jetstream::Settings::Flush();
        std::error_code ec;
        std::filesystem::remove_all(directory, ec);
    }

    SettingsSandbox(const SettingsSandbox&) = delete;
    SettingsSandbox& operator=(const SettingsSandbox&) = delete;

    void require() const {
        if (!Supported) {
            SKIP("Settings isolation is unavailable on this platform");
        }
        REQUIRE(redirected);
    }

    const std::filesystem::path& root() const {
        return directory;
    }

 private:
    static std::filesystem::path Directory(const std::string& label) {
        static std::atomic<unsigned long long> sequence{0};
        const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
        return std::filesystem::temp_directory_path() /
               Jetstream::Platform::PathFromUtf8("cyberether-" + label + "-" + std::to_string(nonce) + "-" +
                                                 std::to_string(sequence.fetch_add(1)));
    }

    std::filesystem::path directory;
    bool redirected = false;
#if defined(JST_OS_LINUX)
    ScopedEnvironment home{"HOME"};
    ScopedEnvironment config{"XDG_CONFIG_HOME"};
#elif defined(JST_OS_WINDOWS)
    ScopedEnvironment appData{"APPDATA"};
#elif defined(JST_OS_MAC)
    ScopedEnvironment home{"CFFIXED_USER_HOME"};
#endif
};

}  // namespace TestSupport

#endif  // JETSTREAM_TESTS_SUPPORT_ENVIRONMENT_HH
