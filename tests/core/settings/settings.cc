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

#include "environment.hh"
#include "jetstream/compositor.hh"
#include "jetstream/logger.hh"
#include "jetstream/platform.hh"
#include "jetstream/settings.hh"

using namespace Jetstream;

namespace {

using ScopedEnvVar = TestSupport::ScopedEnvironment;

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
        , appDataEnv("APPDATA")
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
        if (!appDataEnv.set(Platform::PathToUtf8(tempRoot.root / "AppData" / "Roaming"))) {
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
    ScopedEnvVar appDataEnv;
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

Parser::Map ReadDocument(const std::filesystem::path& path) {
    Parser::Map document;
    if (Parser::YamlDecode(ReadFile(path), document) != Result::SUCCESS) {
        throw std::runtime_error("failed to decode settings file");
    }
    return document;
}

const Parser::Map& Section(const Parser::Map& map, const std::string& key) {
    return std::any_cast<const Parser::Map&>(map.at(key));
}

const Parser::Sequence& List(const Parser::Map& map, const std::string& key) {
    return std::any_cast<const Parser::Sequence&>(map.at(key));
}

std::string Scalar(const Parser::Map& map, const std::string& key) {
    return std::any_cast<const std::string&>(map.at(key));
}

std::string Scalar(const Parser::Sequence& sequence, std::size_t index) {
    return std::any_cast<const std::string&>(sequence.at(index));
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
    REQUIRE(settings.remote.broker.address == "0.0.0.0");
    REQUIRE(settings.remote.broker.port == 8686);
    REQUIRE(settings.remote.broker.publicUrl.empty());
    REQUIRE(settings.remote.broker.certificate.empty());
    REQUIRE(settings.remote.broker.key.empty());
    REQUIRE(settings.remote.video.codec == "h264");
    REQUIRE(settings.remote.video.encoder == "auto");
    REQUIRE_FALSE(settings.remote.access.autoApprove);
    REQUIRE(settings.remote.video.framerate == 30);
    REQUIRE(settings.interface.themeKey == "Dark");
    REQUIRE(settings.interface.infoPanelEnabled);
    REQUIRE(settings.interface.backgroundParticles);
    REQUIRE(settings.intelligence.provider.empty());
    REQUIRE(settings.intelligence.model.empty());
    REQUIRE(settings.intelligence.models.empty());
    REQUIRE(settings.intelligence.customProviders.empty());
    REQUIRE(settings.developer.logLevel == defaults.developer.logLevel);
    REQUIRE_FALSE(settings.developer.latencyEnabled);
    REQUIRE_FALSE(settings.developer.timingEnabled);
    REQUIRE(settings.benchmark.format == "markdown");
    REQUIRE(settings.registry.plugins.empty());
    REQUIRE(settings.runtime.python.path.empty());
    REQUIRE(settings.runtime.dependencyPolicy == "prompt");
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
    settings.remote.broker.publicUrl = "https://example.com";
    settings.remote.video.codec = "av1";
    settings.remote.video.encoder = "software";
    settings.remote.access.autoApprove = true;
    settings.remote.video.framerate = 24;
    settings.developer.logLevel = 4;
    settings.developer.latencyEnabled = true;
    settings.developer.timingEnabled = true;
    settings.registry.plugins.push_back(pluginPath);
    settings.runtime.python.path = runtimePath;
    settings.runtime.dependencyPolicy = "allow";

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
    REQUIRE(restored.remote.broker.publicUrl == "https://example.com");
    REQUIRE(restored.remote.video.codec == "av1");
    REQUIRE(restored.remote.video.encoder == "software");
    REQUIRE(restored.remote.access.autoApprove);
    REQUIRE(restored.remote.video.framerate == 24);
    REQUIRE(restored.developer.logLevel == 4);
    REQUIRE(restored.developer.latencyEnabled);
    REQUIRE(restored.developer.timingEnabled);
    REQUIRE(restored.benchmark.format == "markdown");
    REQUIRE(restored.registry.plugins == settings.registry.plugins);
    REQUIRE(restored.runtime.python.path == runtimePath);
    REQUIRE(restored.runtime.dependencyPolicy == "allow");
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
              "  broker:\n"
              "    publicUrl: https://example.net\n"
              "  access:\n"
              "    autoApprove: true\n"
              "developer:\n"
              "  logLevel: 4\n"
              "registry:\n"
              "  plugins:\n"
              "    - /tmp/cyberether-extra.cep\n"
              "runtime:\n"
              "  dependencyPolicy: deny\n"
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
    REQUIRE(settings.remote.broker.publicUrl == "https://example.net");
    REQUIRE(settings.remote.access.autoApprove);
    REQUIRE(settings.remote.video.codec == "h264");
    REQUIRE(settings.developer.logLevel == 4);
    REQUIRE(settings.registry.plugins.size() == 1);
    REQUIRE(settings.registry.plugins[0] == "/tmp/cyberether-extra.cep");
    REQUIRE(settings.runtime.python.path == "/opt/python/lib/libpython3.12.so");
    REQUIRE(settings.runtime.dependencyPolicy == "deny");
    REQUIRE(settings.benchmark.format == "markdown");
}

TEST_CASE("Settings can update memory without persisting", "[core][settings][persistence]") {
    SettingsSandbox sandbox("memory-only");

    Settings settings;
    settings.interface.themeKey = "Transient";
    settings.developer.timingEnabled = true;
    settings.runtime.python.path = "/runtime/only/libpython.so";
    settings.runtime.dependencyPolicy = "allow";

    REQUIRE(Settings::Set(settings, false) == Result::SUCCESS);
    REQUIRE_FALSE(std::filesystem::exists(sandbox.path));

    Settings restored;
    REQUIRE(Settings::Get(restored) == Result::SUCCESS);
    REQUIRE(restored.interface.themeKey == "Transient");
    REQUIRE(restored.developer.timingEnabled);
    REQUIRE(restored.runtime.python.path == "/runtime/only/libpython.so");
    REQUIRE(restored.runtime.dependencyPolicy == "allow");
}

TEST_CASE("Queued settings publish immediately and preserve persistence ordering", "[core][settings][persistence]") {
    SettingsSandbox sandbox("async");
    Settings settings;
    for (int i = 0; i < 64; ++i) {
        settings.interface.themeKey = "Queued " + std::to_string(i);
        REQUIRE(Settings::SetAsync(settings) == Result::SUCCESS);
        Settings visible;
        REQUIRE(Settings::Get(visible) == Result::SUCCESS);
        CHECK(visible.interface.themeKey == settings.interface.themeKey);
    }
    REQUIRE(Settings::Flush() == Result::SUCCESS);
    CHECK(ReadFile(sandbox.path).find("Queued 63") != std::string::npos);
    settings.interface.themeKey = "Before barrier";
    REQUIRE(Settings::SetAsync(settings) == Result::SUCCESS);
    settings.interface.themeKey = "Synchronous barrier";
    REQUIRE(Settings::Set(settings) == Result::SUCCESS);
    CHECK(ReadFile(sandbox.path).find("Synchronous barrier") != std::string::npos);
    CHECK_FALSE(Settings::TakePersistenceError());
}

TEST_CASE("Queued settings failures reach the owner without reverting newer memory", "[core][settings][persistence]") {
    SettingsSandbox sandbox("async-failure");
    std::filesystem::create_directories(sandbox.path);
    Settings settings;
    settings.interface.themeKey = "Visible despite disk failure";
    REQUIRE(Settings::SetAsync(settings) == Result::SUCCESS);
    REQUIRE(Settings::Flush() == Result::ERROR);
    const auto error = Settings::TakePersistenceError();
    REQUIRE(error);
    CHECK(error->find("Could not save settings") != std::string::npos);
    Settings visible;
    REQUIRE(Settings::Get(visible) == Result::SUCCESS);
    CHECK(visible.interface.themeKey == settings.interface.themeKey);
    CHECK_FALSE(Settings::TakePersistenceError());
}

TEST_CASE("Compositor shutdown reports failed async settings saves without aborting cleanup",
          "[core][settings][persistence][compositor][shutdown]") {
    SettingsSandbox sandbox("shutdown-failure");
    std::filesystem::create_directories(sandbox.path);
    Compositor compositor{CompositorType::DEFAULT};

    JST_LOG_LAST_ERROR().clear();
    REQUIRE(Settings::SetAsync(Settings{}) == Result::SUCCESS);

    SECTION("shutdown drains the queued save") {
        // Let destroy() wait for the failed write rather than flushing it here.
    }

    SECTION("the async failure was already reported before shutdown") {
        REQUIRE(Settings::Flush() == Result::ERROR);
        REQUIRE(Settings::TakePersistenceError());
    }

    // Instance::destroy() checks this result before tearing down its resources.
    CHECK(compositor.destroy() == Result::SUCCESS);
    CHECK(JST_LOG_LAST_ERROR() ==
          "[COMPOSITOR_IMPL_DEFAULT] Failed to save settings during shutdown. Continuing cleanup.");
    CHECK(Settings::Flush() == Result::ERROR);

    while (Settings::TakePersistenceError()) {}
}

TEST_CASE("Transient settings can be restored before a retained update",
          "[core][settings][persistence]") {
    SettingsSandbox sandbox("transient-restore");

    Settings retained;
    retained.graphics.scale = 1.25f;
    retained.remote.broker.publicUrl = "https://retained.example.com";
    REQUIRE(Settings::Set(retained) == Result::SUCCESS);

    Settings runtime = retained;
    runtime.graphics.scale = 3.0f;
    runtime.remote.broker.publicUrl = "https://runtime.example.com";
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

TEST_CASE("Settings preserves unknown sections and nested fields when saving",
          "[core][settings][persistence][compatibility]") {
    SettingsSandbox sandbox("preserve-unknown");
    WriteFile(sandbox.path,
              "graphics:\n"
              "  scale: 1.5\n"
              "  vsync: adaptive\n"
              "  size:\n"
              "    width: 640\n"
              "    density: 2\n"
              "interface:\n"
              "  themeKey: Dark\n"
              "  accent: teal\n"
              "runtime:\n"
              "  dependencyPolicy: deny\n"
              "  python:\n"
              "    path: /opt/python/lib/libpython3.12.so\n"
              "    venv: /opt/venv\n"
              "telemetry:\n"
              "  enabled: true\n"
              "  endpoints:\n"
              "    - https://a.example\n"
              "    - https://b.example\n"
              "  window:\n"
              "    days: 7\n");

    Settings settings;
    REQUIRE(Settings::Get(settings) == Result::SUCCESS);
    settings.interface.themeKey = "Light";
    REQUIRE(Settings::Set(settings) == Result::SUCCESS);

    const auto requirePreserved = [&](const std::string& themeKey) {
        const auto document = ReadDocument(sandbox.path);

        const auto& graphics = Section(document, "graphics");
        CHECK(Scalar(graphics, "vsync") == "adaptive");
        CHECK(Scalar(graphics, "scale") == "1.5");
        CHECK(Scalar(Section(graphics, "size"), "density") == "2");
        CHECK(Scalar(Section(graphics, "size"), "width") == "640");

        const auto& interface = Section(document, "interface");
        CHECK(Scalar(interface, "accent") == "teal");
        CHECK(Scalar(interface, "themeKey") == themeKey);

        const auto& python = Section(Section(document, "runtime"), "python");
        CHECK(Scalar(python, "venv") == "/opt/venv");
        CHECK(Scalar(python, "path") == "/opt/python/lib/libpython3.12.so");

        const auto& telemetry = Section(document, "telemetry");
        CHECK(Scalar(telemetry, "enabled") == "true");
        const auto& endpoints = List(telemetry, "endpoints");
        REQUIRE(endpoints.size() == 2);
        CHECK(Scalar(endpoints, 0) == "https://a.example");
        CHECK(Scalar(endpoints, 1) == "https://b.example");
        CHECK(Scalar(Section(telemetry, "window"), "days") == "7");
    };

    requirePreserved("Light");

    settings.interface.themeKey = "Queued";
    REQUIRE(Settings::SetAsync(settings) == Result::SUCCESS);
    REQUIRE(Settings::Flush() == Result::SUCCESS);
    requirePreserved("Queued");
    CHECK_FALSE(Settings::TakePersistenceError());

    Settings restored;
    REQUIRE(Settings::Get(restored) == Result::SUCCESS);
    CHECK(restored.interface.themeKey == "Queued");
    CHECK(restored.graphics.size.width == 640);
}

TEST_CASE("Settings removes cleared fields without touching unknown siblings",
          "[core][settings][persistence][compatibility]") {
    SettingsSandbox sandbox("clear-known");
    WriteFile(sandbox.path,
              "graphics:\n"
              "  device: cpu\n"
              "  vsync: adaptive\n"
              "registry:\n"
              "  plugins:\n"
              "    - /plugins/a.cep\n"
              "    - /plugins/b.cep\n"
              "  mirror: https://plugins.example\n"
              "runtime:\n"
              "  python:\n"
              "    path: /opt/python/lib/libpython3.12.so\n");

    Settings settings;
    REQUIRE(Settings::Get(settings) == Result::SUCCESS);
    REQUIRE(settings.graphics.device == DeviceType::CPU);

    settings.graphics.device.reset();
    settings.registry.plugins = {"/plugins/a.cep"};
    settings.runtime.python.path.clear();
    REQUIRE(Settings::Set(settings) == Result::SUCCESS);

    const auto document = ReadDocument(sandbox.path);
    const auto& graphics = Section(document, "graphics");
    CHECK_FALSE(graphics.contains("device"));
    CHECK(Scalar(graphics, "vsync") == "adaptive");

    const auto& registry = Section(document, "registry");
    const auto& plugins = List(registry, "plugins");
    REQUIRE(plugins.size() == 1);
    CHECK(Scalar(plugins, 0) == "/plugins/a.cep");
    CHECK(Scalar(registry, "mirror") == "https://plugins.example");

    {
        SettingsSandbox cacheSwitch("clear-known-cache-switch");
        Settings ignored;
        REQUIRE(Settings::Get(ignored) == Result::SUCCESS);
    }

    Settings restored;
    REQUIRE(Settings::Get(restored) == Result::SUCCESS);
    CHECK_FALSE(restored.graphics.device.has_value());
    CHECK(restored.registry.plugins == settings.registry.plugins);
    CHECK(restored.runtime.python.path.empty());
}

TEST_CASE("Settings refuses to overwrite a malformed file",
          "[core][settings][corruption]") {
    SettingsSandbox sandbox("malformed-save");
    const std::string malformed = "graphics: [unterminated\nfuture: value\n";
    WriteFile(sandbox.path, malformed);

    Settings settings;
    settings.interface.themeKey = "Rejected";
    REQUIRE(Settings::Set(settings) == Result::ERROR);
    CHECK(ReadFile(sandbox.path) == malformed);

    REQUIRE(Settings::SetAsync(settings) == Result::SUCCESS);
    REQUIRE(Settings::Flush() == Result::ERROR);
    CHECK(Settings::TakePersistenceError());
    CHECK(ReadFile(sandbox.path) == malformed);
    CHECK_FALSE(std::filesystem::exists(TempSettingsPath(sandbox.path)));
}

TEST_CASE("Settings repairs invalid known values while keeping unknown fields",
          "[core][settings][corruption][compatibility]") {
    SettingsSandbox sandbox("repair-known");
    WriteFile(sandbox.path,
              "graphics: not-a-map\n"
              "future:\n"
              "  flag: on\n");

    Settings rejected;
    REQUIRE(Settings::Get(rejected) == Result::ERROR);

    REQUIRE(Settings::Set(Settings{}) == Result::SUCCESS);

    const auto document = ReadDocument(sandbox.path);
    CHECK(Scalar(Section(document, "graphics"), "framerate") == "60");
    CHECK(Scalar(Section(document, "future"), "flag") == "on");

    Settings restored;
    REQUIRE(Settings::Get(restored) == Result::SUCCESS);
    RequireDefaults(restored);
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
    ScopedEnvVar appDataEnv("APPDATA");
    REQUIRE(appDataEnv.set(std::nullopt));

    Settings settings;
    REQUIRE(Settings::Get(settings) == Result::ERROR);
}

#endif

TEST_CASE("Map rendering preferences preserve old defaults and persist each layer", "[core][settings][maps]") {
    SettingsSandbox sandbox("map-rendering");
    WriteFile(sandbox.path, "maps:\n  highResolution: false\n  futureOption: retained\n");
    Settings settings;
    REQUIRE(Settings::Get(settings) == Result::SUCCESS);
    CHECK_FALSE(settings.maps.highResolution);
    CHECK(settings.maps.rendering == Render::Components::MapRenderingOptions{});

    settings.maps.rendering = {
        .buildings = false,
        .buildingOutlines = false,
        .roads = false,
        .rivers = false,
        .borders = false,
        .places = false,
        .airports = false,
        .bathymetry = false,
    };
    REQUIRE(Settings::SetAsync(settings) == Result::SUCCESS);
    REQUIRE(Settings::Flush() == Result::SUCCESS);
    Settings restored;
    const auto document = ReadDocument(sandbox.path);
    REQUIRE(restored.deserialize(document) == Result::SUCCESS);
    CHECK(restored.maps.rendering == settings.maps.rendering);
    CHECK_FALSE(restored.maps.highResolution);
    CHECK(Scalar(Section(document, "maps"), "futureOption") == "retained");

    settings.maps.rendering.buildings = true;
    settings.maps.rendering.rivers = true;
    settings.maps.rendering.places = true;
    REQUIRE(Settings::SetAsync(settings) == Result::SUCCESS);
    REQUIRE(Settings::Flush() == Result::SUCCESS);
    Settings mixed;
    REQUIRE(mixed.deserialize(ReadDocument(sandbox.path)) == Result::SUCCESS);
    CHECK(mixed.maps.rendering == settings.maps.rendering);
}
