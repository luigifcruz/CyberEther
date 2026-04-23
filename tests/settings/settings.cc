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

struct WindowConfig {
    U64 width = 0;
    U64 height = 0;

    JST_SERDES(width, height);
};

struct MainAppConfig {
    U64 gain = 0;
    bool enabled = false;
    WindowConfig window;

    JST_SERDES(gain, enabled, window);
};

struct RemoteConfig {
    std::string broker;
    U64 framerate = 0;

    JST_SERDES(broker, framerate);
};

std::string ReadFile(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return {};
    }

    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

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
                              std::filesystem::path(L"Roaming-\u00dcnicode")).wstring())) {
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

}  // namespace

TEST_CASE("Settings leaves missing values untouched", "[settings]") {
    SettingsSandbox sandbox("missing");

    MainAppConfig config;
    config.gain = 19;
    config.enabled = true;
    config.window.width = 1280;
    config.window.height = 720;

    REQUIRE(Settings::Load("mainappconf", config) == Result::SUCCESS);
    REQUIRE(config.gain == 19);
    REQUIRE(config.enabled);
    REQUIRE(config.window.width == 1280);
    REQUIRE(config.window.height == 720);
    REQUIRE(!std::filesystem::exists(sandbox.path));
}

TEST_CASE("Settings round-trip serialized structs through YAML", "[settings]") {
    SettingsSandbox sandbox("roundtrip");

    MainAppConfig source;
    source.gain = 42;
    source.enabled = true;
    source.window.width = 1920;
    source.window.height = 1080;

    REQUIRE(Settings::Save("mainappconf", source) == Result::SUCCESS);
    REQUIRE(std::filesystem::exists(sandbox.path));

    MainAppConfig restored;
    REQUIRE(Settings::Load("mainappconf", restored) == Result::SUCCESS);
    REQUIRE(restored.gain == 42);
    REQUIRE(restored.enabled);
    REQUIRE(restored.window.width == 1920);
    REQUIRE(restored.window.height == 1080);

    const auto yaml = ReadFile(sandbox.path);
    REQUIRE(yaml.find("mainappconf:") != std::string::npos);
    REQUIRE(yaml.find("gain: 42") != std::string::npos);
    REQUIRE(yaml.find("window:") != std::string::npos);
}

TEST_CASE("Settings rejects empty keys", "[settings]") {
    SettingsSandbox sandbox("empty-key");

    MainAppConfig config;
    config.gain = 42;
    config.enabled = true;

    REQUIRE(Settings::Save("", config) == Result::ERROR);

    config.gain = 99;
    REQUIRE(Settings::Load("", config) == Result::ERROR);
    REQUIRE(config.gain == 99);

    REQUIRE(Settings::Erase("") == Result::ERROR);
    REQUIRE(!std::filesystem::exists(sandbox.path));
}

TEST_CASE("Settings preserves unrelated keys and erases targeted entries", "[settings]") {
    SettingsSandbox sandbox("merge-erase");

    MainAppConfig mainConfig;
    mainConfig.gain = 7;
    mainConfig.enabled = false;
    mainConfig.window.width = 800;
    mainConfig.window.height = 600;

    RemoteConfig remoteConfig;
    remoteConfig.broker = "https://cyberether.org";
    remoteConfig.framerate = 30;

    REQUIRE(Settings::Save("mainappconf", mainConfig) == Result::SUCCESS);
    REQUIRE(Settings::Save("remote", remoteConfig) == Result::SUCCESS);

    mainConfig.gain = 11;
    mainConfig.window.width = 1440;
    REQUIRE(Settings::Save("mainappconf", mainConfig) == Result::SUCCESS);

    RemoteConfig restoredRemote;
    REQUIRE(Settings::Load("remote", restoredRemote) == Result::SUCCESS);
    REQUIRE(restoredRemote.broker == "https://cyberether.org");
    REQUIRE(restoredRemote.framerate == 30);

    REQUIRE(Settings::Erase("mainappconf") == Result::SUCCESS);

    MainAppConfig missing;
    missing.gain = 99;
    REQUIRE(Settings::Load("mainappconf", missing) == Result::SUCCESS);
    REQUIRE(missing.gain == 99);

    restoredRemote = {};
    REQUIRE(Settings::Load("remote", restoredRemote) == Result::SUCCESS);
    REQUIRE(restoredRemote.broker == "https://cyberether.org");
    REQUIRE(restoredRemote.framerate == 30);

    REQUIRE(Settings::Erase("remote") == Result::SUCCESS);
    REQUIRE(!std::filesystem::exists(sandbox.path));
}
