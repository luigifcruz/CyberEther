#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
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

struct ScopedEnvVar {
    explicit ScopedEnvVar(const char* name) : name(name) {
        if (const char* value = std::getenv(name)) {
            originalValue = value;
        }
    }

    ~ScopedEnvVar() {
        (void)SetEnvValue(name.c_str(), originalValue);
    }

    bool set(const std::optional<std::string>& value) const {
        return SetEnvValue(name.c_str(), value);
    }

    std::string name;
    std::optional<std::string> originalValue;
};

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
        (void)SetWideEnvValue(name.c_str(), originalValue);
    }

    bool set(const std::optional<std::wstring>& value) const {
        return SetWideEnvValue(name.c_str(), value);
    }

    std::wstring name;
    std::optional<std::wstring> originalValue;
};

#endif

struct TempPathRoot {
    explicit TempPathRoot(const std::string& label) {
        static std::atomic<unsigned long long> sequence{0};
        const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
        root = std::filesystem::temp_directory_path() /
               Platform::PathFromUtf8("cyberether-settings-" + label + "-" +
                                      std::to_string(nonce) + "-" +
                                      std::to_string(sequence.fetch_add(1)));
    }

    ~TempPathRoot() {
        std::error_code ec;
        std::filesystem::remove_all(root, ec);
    }

    std::filesystem::path root;
};

struct SettingsSandbox {
    explicit SettingsSandbox(const std::string& label)
        : tempRoot(label)
#if defined(JST_OS_LINUX)
        , homeEnv("HOME")
        , xdgConfigEnv("XDG_CONFIG_HOME")
#elif defined(JST_OS_WINDOWS)
        , appDataEnv(L"APPDATA")
#elif defined(JST_OS_MAC)
        , fixedHomeEnv("CFFIXED_USER_HOME")
#endif
    {
#if defined(JST_OS_LINUX)
        if (!homeEnv.set(Platform::PathToUtf8(tempRoot.root / "home")) ||
            !xdgConfigEnv.set(Platform::PathToUtf8(tempRoot.root / "config"))) {
            throw std::runtime_error("failed to redirect settings test environment");
        }
#elif defined(JST_OS_WINDOWS)
        if (!appDataEnv.set((tempRoot.root / "AppData" / "Roaming").wstring())) {
            throw std::runtime_error("failed to redirect settings test environment");
        }
#elif defined(JST_OS_MAC)
        if (!fixedHomeEnv.set(Platform::PathToUtf8(tempRoot.root))) {
            throw std::runtime_error("failed to redirect settings test environment");
        }
#endif

        std::string configPath;
        if (Platform::ConfigPath(configPath) != Result::SUCCESS) {
            throw std::runtime_error("failed to resolve settings path");
        }

        path = Platform::PathFromUtf8(configPath) / "settings.yaml";
    }

    std::filesystem::path expectedPath() const {
#if defined(JST_OS_LINUX)
        return tempRoot.root / "config" / "cyberether" / "settings.yaml";
#elif defined(JST_OS_WINDOWS)
        return tempRoot.root / "AppData" / "Roaming" / "CyberEther" / "settings.yaml";
#elif defined(JST_OS_MAC)
        return tempRoot.root / "Library" / "Application Support" / "CyberEther" /
               "settings.yaml";
#else
        return path;
#endif
    }

    TempPathRoot tempRoot;
    std::filesystem::path path;

#if defined(JST_OS_LINUX)
    ScopedEnvVar homeEnv;
    ScopedEnvVar xdgConfigEnv;
#elif defined(JST_OS_WINDOWS)
    ScopedWideEnvVar appDataEnv;
#elif defined(JST_OS_MAC)
    ScopedEnvVar fixedHomeEnv;
#endif
};

std::string ReadFile(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return {};
    }

    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

std::filesystem::path TempSettingsPath(const std::filesystem::path& path) {
    auto tempPath = path;
    tempPath += ".tmp";
    return tempPath;
}

void WriteFile(const std::filesystem::path& path, const std::string& content) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream file(path, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!file) {
        throw std::runtime_error("failed to create settings file");
    }

    file.write(content.data(), static_cast<std::streamsize>(content.size()));
    if (!file) {
        throw std::runtime_error("failed to write settings file");
    }
}

void RequireDefaults(const Settings& settings) {
    const Settings defaults;

    REQUIRE_FALSE(settings.graphics.device.has_value());
    REQUIRE(settings.graphics.deviceId == 0);
    REQUIRE_FALSE(settings.graphics.headless);
    REQUIRE(settings.graphics.size.width == 1920);
    REQUIRE(settings.graphics.size.height == 1080);
    REQUIRE(settings.graphics.scale == 1.0f);
    REQUIRE(settings.graphics.framerate == 60);
    REQUIRE(settings.remote.brokerUrl == "https://cyberether.org");
    REQUIRE(settings.remote.codec == "h264");
    REQUIRE(settings.remote.encoder == "auto");
    REQUIRE_FALSE(settings.remote.autoJoinSessions);
    REQUIRE(settings.remote.framerate == 30);
    REQUIRE(settings.interface.themeKey == "Dark");
    REQUIRE(settings.interface.infoPanelEnabled);
    REQUIRE(settings.interface.backgroundParticles);
    REQUIRE(settings.developer.logLevel == defaults.developer.logLevel);
    REQUIRE_FALSE(settings.developer.latencyEnabled);
    REQUIRE_FALSE(settings.developer.timingEnabled);
    REQUIRE(settings.benchmark.format == "markdown");
    REQUIRE(settings.registry.plugins.empty());
    REQUIRE(settings.runtime.python.path.empty());
}

}  // namespace

TEST_CASE("Settings returns complete defaults when its file is missing",
          "[core][settings][defaults]") {
    SettingsSandbox sandbox("missing");
    REQUIRE(sandbox.path == sandbox.expectedPath());

    Settings settings;
    REQUIRE(Settings::Get(settings) == Result::SUCCESS);

    RequireDefaults(settings);
    REQUIRE_FALSE(std::filesystem::exists(sandbox.path));
    REQUIRE_FALSE(std::filesystem::exists(TempSettingsPath(sandbox.path)));
}

TEST_CASE("Settings treats an empty file as defaults", "[core][settings][defaults]") {
    SettingsSandbox sandbox("empty");
    WriteFile(sandbox.path, "");

    Settings settings;
    REQUIRE(Settings::Get(settings) == Result::SUCCESS);
    RequireDefaults(settings);
}

TEST_CASE("Settings persists serialized fields and reloads them from disk",
          "[core][settings][persistence]") {
    SettingsSandbox sandbox("roundtrip");
    const auto runtimePath = Platform::PathToUtf8(
        sandbox.tempRoot.root / "Python Runtime" / "libpython3.14.dylib");
    const auto pluginPath = Platform::PathToUtf8(sandbox.tempRoot.root / "plugins" / "extra.cep");

    Settings settings;
    settings.benchmark.format = "json";
    settings.graphics.deviceId = 2;
    settings.graphics.headless = true;
    settings.graphics.size.width = 1280;
    settings.graphics.size.height = 720;
    settings.graphics.scale = 1.25f;
    settings.graphics.framerate = 75;
    settings.interface.themeKey = "Light";
    settings.interface.infoPanelEnabled = false;
    settings.interface.backgroundParticles = false;
    settings.remote.brokerUrl = "https://example.com";
    settings.remote.codec = "av1";
    settings.remote.encoder = "software";
    settings.remote.autoJoinSessions = true;
    settings.remote.framerate = 24;
    settings.developer.logLevel = 4;
    settings.developer.latencyEnabled = true;
    settings.developer.timingEnabled = true;
    settings.registry.plugins.push_back(pluginPath);
    settings.runtime.python.path = runtimePath;

    REQUIRE(Settings::Set(settings) == Result::SUCCESS);
    REQUIRE(std::filesystem::exists(sandbox.path));
    REQUIRE_FALSE(std::filesystem::exists(TempSettingsPath(sandbox.path)));

    const auto yaml = ReadFile(sandbox.path);
    REQUIRE(yaml.find("benchmark:") == std::string::npos);
    REQUIRE(yaml.find("format:") == std::string::npos);
    REQUIRE(yaml.find("headless:") == std::string::npos);
    REQUIRE(yaml.find("deviceId:") == std::string::npos);
    REQUIRE(yaml.find("graphics:") != std::string::npos);
    REQUIRE(yaml.find("interface:") != std::string::npos);
    REQUIRE(yaml.find("remote:") != std::string::npos);
    REQUIRE(yaml.find("developer:") != std::string::npos);
    REQUIRE(yaml.find("registry:") != std::string::npos);
    REQUIRE(yaml.find("runtime:") != std::string::npos);
    REQUIRE(yaml.find("python:") != std::string::npos);

    {
        SettingsSandbox cacheSwitch("roundtrip-cache-switch");
        Settings ignored;
        REQUIRE(Settings::Get(ignored) == Result::SUCCESS);
    }

    Settings restored;
    REQUIRE(Settings::Get(restored) == Result::SUCCESS);
    REQUIRE_FALSE(restored.graphics.device.has_value());
    REQUIRE(restored.graphics.deviceId == 0);
    REQUIRE_FALSE(restored.graphics.headless);
    REQUIRE(restored.graphics.size.width == 1280);
    REQUIRE(restored.graphics.size.height == 720);
    REQUIRE(restored.graphics.scale == 1.25f);
    REQUIRE(restored.graphics.framerate == 75);
    REQUIRE(restored.interface.themeKey == "Light");
    REQUIRE_FALSE(restored.interface.infoPanelEnabled);
    REQUIRE_FALSE(restored.interface.backgroundParticles);
    REQUIRE(restored.remote.brokerUrl == "https://example.com");
    REQUIRE(restored.remote.codec == "av1");
    REQUIRE(restored.remote.encoder == "software");
    REQUIRE(restored.remote.autoJoinSessions);
    REQUIRE(restored.remote.framerate == 24);
    REQUIRE(restored.developer.logLevel == 4);
    REQUIRE(restored.developer.latencyEnabled);
    REQUIRE(restored.developer.timingEnabled);
    REQUIRE(restored.benchmark.format == "markdown");
    REQUIRE(restored.registry.plugins == settings.registry.plugins);
    REQUIRE(restored.runtime.python.path == runtimePath);
}

TEST_CASE("Settings loads partial YAML over defaults", "[core][settings][persistence]") {
    SettingsSandbox sandbox("existing");
    WriteFile(sandbox.path,
              "graphics:\n"
              "  headless: true\n"
              "  deviceId: 9\n"
              "  size:\n"
              "    width: 640\n"
              "  scale: 1.5\n"
              "benchmark:\n"
              "  format: csv\n"
              "interface:\n"
              "  themeKey: Solarized\n"
              "  infoPanelEnabled: false\n"
              "remote:\n"
              "  brokerUrl: https://example.net\n"
              "  autoJoinSessions: true\n"
              "developer:\n"
              "  logLevel: 4\n"
              "registry:\n"
              "  plugins:\n"
              "    - /tmp/cyberether-extra.cep\n"
              "runtime:\n"
              "  python:\n"
              "    path: /opt/python/lib/libpython3.12.so\n"
              "unknown: ignored\n");

    Settings settings;
    REQUIRE(Settings::Get(settings) == Result::SUCCESS);

    REQUIRE(settings.graphics.size.width == 640);
    REQUIRE(settings.graphics.size.height == 1080);
    REQUIRE(settings.graphics.scale == 1.5f);
    REQUIRE(settings.graphics.framerate == 60);
    REQUIRE_FALSE(settings.graphics.headless);
    REQUIRE(settings.graphics.deviceId == 0);
    REQUIRE(settings.interface.themeKey == "Solarized");
    REQUIRE_FALSE(settings.interface.infoPanelEnabled);
    REQUIRE(settings.interface.backgroundParticles);
    REQUIRE(settings.remote.brokerUrl == "https://example.net");
    REQUIRE(settings.remote.autoJoinSessions);
    REQUIRE(settings.remote.codec == "h264");
    REQUIRE(settings.developer.logLevel == 4);
    REQUIRE(settings.registry.plugins.size() == 1);
    REQUIRE(settings.registry.plugins[0] == "/tmp/cyberether-extra.cep");
    REQUIRE(settings.runtime.python.path == "/opt/python/lib/libpython3.12.so");
    REQUIRE(settings.benchmark.format == "markdown");
}

TEST_CASE("Settings can update memory without persisting", "[core][settings][persistence]") {
    SettingsSandbox sandbox("memory-only");

    Settings settings;
    settings.interface.themeKey = "Transient";
    settings.developer.timingEnabled = true;
    settings.runtime.python.path = "/runtime/only/libpython.so";

    REQUIRE(Settings::Set(settings, false) == Result::SUCCESS);
    REQUIRE_FALSE(std::filesystem::exists(sandbox.path));

    Settings restored;
    REQUIRE(Settings::Get(restored) == Result::SUCCESS);
    REQUIRE(restored.interface.themeKey == "Transient");
    REQUIRE(restored.developer.timingEnabled);
    REQUIRE(restored.runtime.python.path == "/runtime/only/libpython.so");
}

TEST_CASE("Transient settings can be restored before a retained update",
          "[core][settings][persistence]") {
    SettingsSandbox sandbox("transient-restore");

    Settings retained;
    retained.graphics.scale = 1.25f;
    retained.remote.brokerUrl = "https://retained.example.com";
    REQUIRE(Settings::Set(retained) == Result::SUCCESS);

    Settings runtime = retained;
    runtime.graphics.scale = 3.0f;
    runtime.remote.brokerUrl = "https://runtime.example.com";
    REQUIRE(Settings::Set(runtime, false) == Result::SUCCESS);
    REQUIRE(Settings::Set(retained, false) == Result::SUCCESS);

    Settings updated;
    REQUIRE(Settings::Get(updated) == Result::SUCCESS);
    updated.interface.infoPanelEnabled = false;
    REQUIRE(Settings::Set(updated) == Result::SUCCESS);

    const auto yaml = ReadFile(sandbox.path);
    REQUIRE(yaml.find("https://retained.example.com") != std::string::npos);
    REQUIRE(yaml.find("https://runtime.example.com") == std::string::npos);
    REQUIRE(yaml.find("scale: 1.25") != std::string::npos);
}

TEST_CASE("Settings retries a malformed file after it is repaired",
          "[core][settings][corruption]") {
    SettingsSandbox sandbox("malformed-retry");
    WriteFile(sandbox.path, "graphics: [unterminated\n");

    Settings untouched;
    untouched.interface.themeKey = "Caller value";
    REQUIRE(Settings::Get(untouched) == Result::ERROR);
    REQUIRE(untouched.interface.themeKey == "Caller value");

    WriteFile(sandbox.path,
              "interface:\n"
              "  themeKey: Repaired\n");

    Settings repaired;
    REQUIRE(Settings::Get(repaired) == Result::SUCCESS);
    REQUIRE(repaired.interface.themeKey == "Repaired");
    REQUIRE(repaired.graphics.size.width == 1920);
}

TEST_CASE("Settings rejects structurally corrupt field values",
          "[core][settings][corruption]") {
    SettingsSandbox sandbox("invalid-type");
    WriteFile(sandbox.path, "graphics: not-a-map\n");

    Settings untouched;
    untouched.runtime.python.path = "caller-runtime";
    REQUIRE(Settings::Get(untouched) == Result::ERROR);
    REQUIRE(untouched.runtime.python.path == "caller-runtime");
}

TEST_CASE("A corrupt path does not damage settings cached for another path",
          "[core][settings][corruption]") {
    SettingsSandbox retainedSandbox("retained-before-corruption");
    Settings retained;
    retained.interface.themeKey = "Retained";
    retained.runtime.python.path = "/retained/libpython.so";
    REQUIRE(Settings::Set(retained, false) == Result::SUCCESS);

    {
        SettingsSandbox corruptSandbox("corrupt-other-path");
        WriteFile(corruptSandbox.path, "interface: [unterminated\n");
        Settings untouched;
        REQUIRE(Settings::Get(untouched) == Result::ERROR);
    }

    Settings restored;
    REQUIRE(Settings::Get(restored) == Result::SUCCESS);
    // Expected to fail currently: a failed load resets the settings cached for the prior path.
    REQUIRE(restored.interface.themeKey == "Retained");
    REQUIRE(restored.runtime.python.path == "/retained/libpython.so");
}

TEST_CASE("A failed persistent update preserves the in-memory settings",
          "[core][settings][persistence]") {
    SettingsSandbox sandbox("failed-persistence");
    Settings retained;
    retained.interface.themeKey = "Retained";
    retained.runtime.python.path = "/retained/libpython.so";
    REQUIRE(Settings::Set(retained, false) == Result::SUCCESS);

    std::filesystem::create_directories(sandbox.path);

    Settings rejected = retained;
    rejected.interface.themeKey = "Rejected";
    rejected.runtime.python.path = "/rejected/libpython.so";
    REQUIRE(Settings::Set(rejected) == Result::ERROR);
    REQUIRE_FALSE(std::filesystem::exists(TempSettingsPath(sandbox.path)));

    Settings restored;
    REQUIRE(Settings::Get(restored) == Result::SUCCESS);
    // Expected to fail currently: Set updates the cache before persistence succeeds.
    REQUIRE(restored.interface.themeKey == "Retained");
    REQUIRE(restored.runtime.python.path == "/retained/libpython.so");
}

TEST_CASE("Settings replaces an existing file without leaving a temporary file",
          "[core][settings][persistence]") {
    SettingsSandbox sandbox("replace-existing");
    Settings settings;
    settings.interface.themeKey = "First";
    REQUIRE(Settings::Set(settings) == Result::SUCCESS);

    settings.interface.themeKey = "Second";
#if defined(JST_OS_WINDOWS)
    // Expected to fail currently on Windows: rename does not replace an existing destination.
    REQUIRE(Settings::Set(settings) == Result::SUCCESS);
#else
    REQUIRE(Settings::Set(settings) == Result::SUCCESS);
#endif

    const auto yaml = ReadFile(sandbox.path);
    REQUIRE(yaml.find("themeKey: Second") != std::string::npos);
    REQUIRE(yaml.find("themeKey: First") == std::string::npos);
    REQUIRE_FALSE(std::filesystem::exists(TempSettingsPath(sandbox.path)));
}

#if defined(JST_OS_LINUX)

TEST_CASE("Settings falls back to HOME when XDG_CONFIG_HOME is relative",
          "[core][settings][environment]") {
    TempPathRoot tempRoot("relative-xdg");
    ScopedEnvVar homeEnv("HOME");
    ScopedEnvVar xdgConfigEnv("XDG_CONFIG_HOME");
    REQUIRE(homeEnv.set(Platform::PathToUtf8(tempRoot.root / "home")));
    REQUIRE(xdgConfigEnv.set("relative/config"));

    Settings settings;
    settings.interface.themeKey = "Home fallback";
    REQUIRE(Settings::Set(settings) == Result::SUCCESS);

    const auto expected = tempRoot.root / "home" / ".config" / "cyberether" / "settings.yaml";
    REQUIRE(std::filesystem::exists(expected));
}

TEST_CASE("Settings reports an error when Linux config environment is unavailable",
          "[core][settings][environment]") {
    ScopedEnvVar homeEnv("HOME");
    ScopedEnvVar xdgConfigEnv("XDG_CONFIG_HOME");
    REQUIRE(homeEnv.set(std::nullopt));
    REQUIRE(xdgConfigEnv.set(std::nullopt));

    Settings settings;
    REQUIRE(Settings::Get(settings) == Result::ERROR);
}

#elif defined(JST_OS_WINDOWS)

TEST_CASE("Settings reports an error when Windows config environment is unavailable",
          "[core][settings][environment]") {
    ScopedWideEnvVar appDataEnv(L"APPDATA");
    REQUIRE(appDataEnv.set(std::nullopt));

    Settings settings;
    REQUIRE(Settings::Get(settings) == Result::ERROR);
}

#endif
