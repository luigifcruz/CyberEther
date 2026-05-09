#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

#include "jetstream/platform.hh"
#include "jetstream/settings.hh"

using namespace Jetstream;

namespace {

bool SetEnvValue(const char* name, const std::optional<std::string>& value) {
#if defined(JST_OS_WINDOWS)
    return _putenv_s(name, value ? value->c_str() : "") == 0;
#else
    if (value) {
        return setenv(name, value->c_str(), 1) == 0;
    }

    return unsetenv(name) == 0;
#endif
}

#if defined(JST_OS_WINDOWS)

bool SetWideEnvValue(const wchar_t* name, const std::optional<std::wstring>& value) {
    return _wputenv_s(name, value ? value->c_str() : L"") == 0;
}

struct ScopedWideEnvVar {
    explicit ScopedWideEnvVar(const wchar_t* name) : name(name) {
        if (const wchar_t* value = _wgetenv(name)) {
            originalValue = value;
        }
    }

    ~ScopedWideEnvVar() {
        (void)SetWideEnvValue(name, originalValue);
    }

    bool set(const std::optional<std::wstring>& value) const {
        return SetWideEnvValue(name, value);
    }

    const wchar_t* name;
    std::optional<std::wstring> originalValue;
};

#endif

struct ScopedEnvVar {
    explicit ScopedEnvVar(const char* name) : name(name) {
        if (const char* value = std::getenv(name)) {
            originalValue = value;
        }
    }

    ~ScopedEnvVar() {
        (void)SetEnvValue(name, originalValue);
    }

    bool set(const std::optional<std::string>& value) const {
        return SetEnvValue(name, value);
    }

    const char* name;
    std::optional<std::string> originalValue;
};

struct TempPathRoot {
    explicit TempPathRoot(const std::string& label) {
        const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
        root = std::filesystem::temp_directory_path() /
               ("cyberether-settings-" + label + "-" + std::to_string(nonce));
    }

    ~TempPathRoot() {
        std::error_code ec;
        std::filesystem::remove_all(root, ec);
    }

    std::filesystem::path root;
};

struct SettingsSandbox {
    explicit SettingsSandbox(const std::string& label) {
#if defined(JST_OS_LINUX)
        tempRoot = std::make_unique<TempPathRoot>(label);
        homeEnv = std::make_unique<ScopedEnvVar>("HOME");
        xdgConfigEnv = std::make_unique<ScopedEnvVar>("XDG_CONFIG_HOME");

        if (!homeEnv->set((tempRoot->root / "home").string()) ||
            !xdgConfigEnv->set((tempRoot->root / "config").string())) {
            throw std::runtime_error("failed to redirect settings test environment");
        }
#elif defined(JST_OS_WINDOWS)
        tempRoot = std::make_unique<TempPathRoot>(label);
        appDataEnv = std::make_unique<ScopedWideEnvVar>(L"APPDATA");

        if (!appDataEnv->set((tempRoot->root / "AppData" /
                              std::filesystem::path(L"Roaming-Unicode")).wstring())) {
            throw std::runtime_error("failed to redirect settings test environment");
        }
#elif defined(JST_OS_MAC)
        tempRoot = std::make_unique<TempPathRoot>(label);
        fixedHomeEnv = std::make_unique<ScopedEnvVar>("CFFIXED_USER_HOME");

        if (!fixedHomeEnv->set(tempRoot->root.string())) {
            throw std::runtime_error("failed to redirect settings test environment");
        }
#endif

        std::string configPath;
        if (Platform::ConfigPath(configPath) != Result::SUCCESS) {
            throw std::runtime_error("failed to resolve settings path");
        }

        path = std::filesystem::u8path(configPath) / "settings.yaml";
    }

    ~SettingsSandbox() = default;

    std::filesystem::path path;

 private:
#if defined(JST_OS_LINUX) || defined(JST_OS_WINDOWS) || defined(JST_OS_MAC)
    std::unique_ptr<TempPathRoot> tempRoot;
#endif

#if defined(JST_OS_LINUX)
    std::unique_ptr<ScopedEnvVar> homeEnv;
    std::unique_ptr<ScopedEnvVar> xdgConfigEnv;
#elif defined(JST_OS_WINDOWS)
    std::unique_ptr<ScopedWideEnvVar> appDataEnv;
#elif defined(JST_OS_MAC)
    std::unique_ptr<ScopedEnvVar> fixedHomeEnv;
#endif
};

std::string ReadFile(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return {};
    }

    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

void WriteFile(const std::filesystem::path& path, const std::string& content) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream file(path, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!file) {
        throw std::runtime_error("failed to create settings file");
    }

    file.write(content.data(), static_cast<std::streamsize>(content.size()));
}

}  // namespace

TEST_CASE("Settings returns defaults when file is missing", "[settings]") {
    SettingsSandbox sandbox("missing");

    Settings settings;
    REQUIRE(Settings::Get(settings) == Result::SUCCESS);

    REQUIRE(settings.graphics.size.width == 1920);
    REQUIRE(settings.graphics.size.height == 1080);
    REQUIRE(settings.interface.themeKey == "Dark");
    REQUIRE(settings.interface.infoPanelEnabled);
    REQUIRE(settings.remote.brokerUrl == "https://cyberether.org");
    REQUIRE(!std::filesystem::exists(sandbox.path));
}

TEST_CASE("Settings persists root YAML", "[settings]") {
    SettingsSandbox sandbox("roundtrip");

    Settings settings;
    settings.graphics.size.width = 1280;
    settings.graphics.size.height = 720;
    settings.interface.themeKey = "Light";
    settings.remote.brokerUrl = "https://example.com";
    settings.remote.autoJoinSessions = true;

    REQUIRE(Settings::Set(settings) == Result::SUCCESS);
    REQUIRE(std::filesystem::exists(sandbox.path));

    const auto yaml = ReadFile(sandbox.path);
    REQUIRE(yaml.find("graphics:") != std::string::npos);
    REQUIRE(yaml.find("interface:") != std::string::npos);
    REQUIRE(yaml.find("remote:") != std::string::npos);
    REQUIRE(yaml.find("themeKey: Light") != std::string::npos);
    REQUIRE(yaml.find("brokerUrl:") != std::string::npos);
    REQUIRE(yaml.find("https://example.com") != std::string::npos);
}

TEST_CASE("Settings loads existing YAML", "[settings]") {
    SettingsSandbox sandbox("existing");

    WriteFile(sandbox.path,
              "interface:\n"
              "  themeKey: Solarized\n"
              "  infoPanelEnabled: false\n"
              "remote:\n"
              "  brokerUrl: https://example.net\n"
              "  autoJoinSessions: true\n"
              "developer:\n"
              "  logLevel: 4\n");

    Settings settings;
    REQUIRE(Settings::Get(settings) == Result::SUCCESS);

    REQUIRE(settings.interface.themeKey == "Solarized");
    REQUIRE(!settings.interface.infoPanelEnabled);
    REQUIRE(settings.remote.brokerUrl == "https://example.net");
    REQUIRE(settings.remote.autoJoinSessions);
    REQUIRE(settings.developer.logLevel == 4);
    REQUIRE(settings.graphics.framerate == 60);
}

TEST_CASE("Settings can update memory without persisting", "[settings]") {
    SettingsSandbox sandbox("memory-only");

    Settings settings;
    settings.interface.themeKey = "Transient";
    settings.developer.runtimeMetricsEnabled = true;

    REQUIRE(Settings::Set(settings, false) == Result::SUCCESS);
    REQUIRE(!std::filesystem::exists(sandbox.path));

    Settings restored;
    REQUIRE(Settings::Get(restored) == Result::SUCCESS);
    REQUIRE(restored.interface.themeKey == "Transient");
    REQUIRE(restored.developer.runtimeMetricsEnabled);
}
