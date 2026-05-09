#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>

#include "jetstream/platform.hh"

using namespace Jetstream;

namespace {

constexpr const char* kInitialConfigPath = "existing-config";
constexpr const char* kInitialCachePath = "existing-cache";

void SeedPaths(std::string& configPath, std::string& cachePath) {
    configPath = kInitialConfigPath;
    cachePath = kInitialCachePath;
}

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

std::string PathToUtf8(const std::filesystem::path& path) {
    const auto utf8Path = path.u8string();

    std::string utf8String;
    utf8String.reserve(utf8Path.size());
    for (const auto ch : utf8Path) {
        utf8String.push_back(static_cast<char>(ch));
    }

    return utf8String;
}

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
               ("cyberether-platform-" + label + "-" + std::to_string(nonce));
    }

    ~TempPathRoot() {
        std::error_code ec;
        std::filesystem::remove_all(root, ec);
    }

    std::filesystem::path root;
};

}  // namespace

TEST_CASE("Platform config and cache paths follow platform conventions", "[platform][paths]") {
#if defined(JST_OS_LINUX)
    SECTION("absolute XDG overrides do not require HOME and do not create directories") {
        TempPathRoot temp("linux-absolute");
        const auto configRoot = temp.root / "config-root";
        const auto cacheRoot = temp.root / "cache-root";
        const auto expectedConfig = configRoot / "cyberether";
        const auto expectedCache = cacheRoot / "cyberether";

        const ScopedEnvVar homeEnv("HOME");
        const ScopedEnvVar xdgConfigEnv("XDG_CONFIG_HOME");
        const ScopedEnvVar xdgCacheEnv("XDG_CACHE_HOME");

        REQUIRE(homeEnv.set(std::nullopt));
        REQUIRE(xdgConfigEnv.set(configRoot.string()));
        REQUIRE(xdgCacheEnv.set(cacheRoot.string()));

        std::string configPath;
        std::string cachePath;
        SeedPaths(configPath, cachePath);
        REQUIRE(Platform::ConfigPath(configPath) == Result::SUCCESS);
        REQUIRE(Platform::CachePath(cachePath) == Result::SUCCESS);

        REQUIRE(std::filesystem::path(configPath) == expectedConfig);
        REQUIRE(std::filesystem::path(cachePath) == expectedCache);
        REQUIRE(!std::filesystem::exists(configRoot));
        REQUIRE(!std::filesystem::exists(expectedConfig));
        REQUIRE(!std::filesystem::exists(cacheRoot));
        REQUIRE(!std::filesystem::exists(expectedCache));
    }

    SECTION("XDG unset falls back to HOME and does not create directories") {
        TempPathRoot temp("linux-home");
        const auto homeRoot = temp.root / "home-root";
        const auto expectedConfig = homeRoot / ".config" / "cyberether";
        const auto expectedCache = homeRoot / ".cache" / "cyberether";

        const ScopedEnvVar homeEnv("HOME");
        const ScopedEnvVar xdgConfigEnv("XDG_CONFIG_HOME");
        const ScopedEnvVar xdgCacheEnv("XDG_CACHE_HOME");

        REQUIRE(homeEnv.set(homeRoot.string()));
        REQUIRE(xdgConfigEnv.set(std::nullopt));
        REQUIRE(xdgCacheEnv.set(std::nullopt));

        std::string configPath;
        std::string cachePath;
        SeedPaths(configPath, cachePath);
        REQUIRE(Platform::ConfigPath(configPath) == Result::SUCCESS);
        REQUIRE(Platform::CachePath(cachePath) == Result::SUCCESS);

        REQUIRE(std::filesystem::path(configPath) == expectedConfig);
        REQUIRE(std::filesystem::path(cachePath) == expectedCache);
        REQUIRE(!std::filesystem::exists(homeRoot / ".config"));
        REQUIRE(!std::filesystem::exists(expectedConfig));
        REQUIRE(!std::filesystem::exists(homeRoot / ".cache"));
        REQUIRE(!std::filesystem::exists(expectedCache));
    }

    SECTION("relative XDG overrides fall back to HOME and do not create directories") {
        TempPathRoot temp("linux-relative");
        const auto homeRoot = temp.root / "home-root";
        const auto expectedConfig = homeRoot / ".config" / "cyberether";
        const auto expectedCache = homeRoot / ".cache" / "cyberether";

        const ScopedEnvVar homeEnv("HOME");
        const ScopedEnvVar xdgConfigEnv("XDG_CONFIG_HOME");
        const ScopedEnvVar xdgCacheEnv("XDG_CACHE_HOME");

        REQUIRE(homeEnv.set(homeRoot.string()));
        REQUIRE(xdgConfigEnv.set("relative-config"));
        REQUIRE(xdgCacheEnv.set("relative-cache"));

        std::string configPath;
        std::string cachePath;
        SeedPaths(configPath, cachePath);
        REQUIRE(Platform::ConfigPath(configPath) == Result::SUCCESS);
        REQUIRE(Platform::CachePath(cachePath) == Result::SUCCESS);

        REQUIRE(std::filesystem::path(configPath) == expectedConfig);
        REQUIRE(std::filesystem::path(cachePath) == expectedCache);
        REQUIRE(!std::filesystem::exists(homeRoot / ".config"));
        REQUIRE(!std::filesystem::exists(expectedConfig));
        REQUIRE(!std::filesystem::exists(homeRoot / ".cache"));
        REQUIRE(!std::filesystem::exists(expectedCache));
    }

    SECTION("config can use XDG while cache falls back to HOME") {
        TempPathRoot temp("linux-mixed-config");
        const auto homeRoot = temp.root / "home-root";
        const auto configRoot = temp.root / "config-root";
        const auto expectedConfig = configRoot / "cyberether";
        const auto expectedCache = homeRoot / ".cache" / "cyberether";

        const ScopedEnvVar homeEnv("HOME");
        const ScopedEnvVar xdgConfigEnv("XDG_CONFIG_HOME");
        const ScopedEnvVar xdgCacheEnv("XDG_CACHE_HOME");

        REQUIRE(homeEnv.set(homeRoot.string()));
        REQUIRE(xdgConfigEnv.set(configRoot.string()));
        REQUIRE(xdgCacheEnv.set(std::nullopt));

        std::string configPath;
        std::string cachePath;
        SeedPaths(configPath, cachePath);
        REQUIRE(Platform::ConfigPath(configPath) == Result::SUCCESS);
        REQUIRE(Platform::CachePath(cachePath) == Result::SUCCESS);

        REQUIRE(std::filesystem::path(configPath) == expectedConfig);
        REQUIRE(std::filesystem::path(cachePath) == expectedCache);
    }

    SECTION("cache can use XDG while config falls back to HOME") {
        TempPathRoot temp("linux-mixed-cache");
        const auto homeRoot = temp.root / "home-root";
        const auto cacheRoot = temp.root / "cache-root";
        const auto expectedConfig = homeRoot / ".config" / "cyberether";
        const auto expectedCache = cacheRoot / "cyberether";

        const ScopedEnvVar homeEnv("HOME");
        const ScopedEnvVar xdgConfigEnv("XDG_CONFIG_HOME");
        const ScopedEnvVar xdgCacheEnv("XDG_CACHE_HOME");

        REQUIRE(homeEnv.set(homeRoot.string()));
        REQUIRE(xdgConfigEnv.set(std::nullopt));
        REQUIRE(xdgCacheEnv.set(cacheRoot.string()));

        std::string configPath;
        std::string cachePath;
        SeedPaths(configPath, cachePath);
        REQUIRE(Platform::ConfigPath(configPath) == Result::SUCCESS);
        REQUIRE(Platform::CachePath(cachePath) == Result::SUCCESS);

        REQUIRE(std::filesystem::path(configPath) == expectedConfig);
        REQUIRE(std::filesystem::path(cachePath) == expectedCache);
    }

    SECTION("relative XDG overrides require HOME and fail cleanly without it") {
        const ScopedEnvVar homeEnv("HOME");
        const ScopedEnvVar xdgConfigEnv("XDG_CONFIG_HOME");
        const ScopedEnvVar xdgCacheEnv("XDG_CACHE_HOME");

        REQUIRE(homeEnv.set(std::nullopt));
        REQUIRE(xdgConfigEnv.set("relative-config"));
        REQUIRE(xdgCacheEnv.set("relative-cache"));

        std::string configPath;
        std::string cachePath;
        SeedPaths(configPath, cachePath);
        REQUIRE(Platform::ConfigPath(configPath) == Result::ERROR);
        REQUIRE(Platform::CachePath(cachePath) == Result::ERROR);
        REQUIRE(configPath == kInitialConfigPath);
        REQUIRE(cachePath == kInitialCachePath);
    }

    SECTION("empty XDG overrides are treated as unset") {
        TempPathRoot temp("linux-empty");
        const auto homeRoot = temp.root / "home-root";
        const auto expectedConfig = homeRoot / ".config" / "cyberether";
        const auto expectedCache = homeRoot / ".cache" / "cyberether";

        const ScopedEnvVar homeEnv("HOME");
        const ScopedEnvVar xdgConfigEnv("XDG_CONFIG_HOME");
        const ScopedEnvVar xdgCacheEnv("XDG_CACHE_HOME");

        REQUIRE(homeEnv.set(homeRoot.string()));
        REQUIRE(xdgConfigEnv.set(std::string()));
        REQUIRE(xdgCacheEnv.set(std::string()));

        std::string configPath;
        std::string cachePath;
        SeedPaths(configPath, cachePath);
        REQUIRE(Platform::ConfigPath(configPath) == Result::SUCCESS);
        REQUIRE(Platform::CachePath(cachePath) == Result::SUCCESS);

        REQUIRE(std::filesystem::path(configPath) == expectedConfig);
        REQUIRE(std::filesystem::path(cachePath) == expectedCache);
    }
#elif defined(JST_OS_WINDOWS)
    SECTION("windows resolves APPDATA and LOCALAPPDATA without creating directories") {
        TempPathRoot temp("windows");
        const auto appDataRoot = temp.root / "Roaming";
        const auto localAppDataRoot = temp.root / "Local";
        const auto expectedConfig = appDataRoot / "CyberEther";
        const auto expectedCache = localAppDataRoot / "CyberEther" / "Cache";

        const ScopedEnvVar appDataEnv("APPDATA");
        const ScopedEnvVar localAppDataEnv("LOCALAPPDATA");

        REQUIRE(appDataEnv.set(appDataRoot.string()));
        REQUIRE(localAppDataEnv.set(localAppDataRoot.string()));

        std::string configPath;
        std::string cachePath;
        SeedPaths(configPath, cachePath);
        REQUIRE(Platform::ConfigPath(configPath) == Result::SUCCESS);
        REQUIRE(Platform::CachePath(cachePath) == Result::SUCCESS);

        REQUIRE(configPath == PathToUtf8(expectedConfig));
        REQUIRE(cachePath == PathToUtf8(expectedCache));
        REQUIRE(!std::filesystem::exists(expectedConfig));
        REQUIRE(!std::filesystem::exists(expectedCache));
    }

    SECTION("windows cache falls back to APPDATA when LOCALAPPDATA is unavailable") {
        TempPathRoot temp("windows-fallback");
        const auto appDataRoot = temp.root / "Roaming";
        const auto expectedConfig = appDataRoot / "CyberEther";
        const auto expectedCache = appDataRoot / "CyberEther" / "Cache";

        const ScopedEnvVar appDataEnv("APPDATA");
        const ScopedEnvVar localAppDataEnv("LOCALAPPDATA");

        REQUIRE(appDataEnv.set(appDataRoot.string()));
        REQUIRE(localAppDataEnv.set(std::nullopt));

        std::string configPath;
        std::string cachePath;
        SeedPaths(configPath, cachePath);
        REQUIRE(Platform::ConfigPath(configPath) == Result::SUCCESS);
        REQUIRE(Platform::CachePath(cachePath) == Result::SUCCESS);

        REQUIRE(configPath == PathToUtf8(expectedConfig));
        REQUIRE(cachePath == PathToUtf8(expectedCache));
        REQUIRE(!std::filesystem::exists(expectedConfig));
        REQUIRE(!std::filesystem::exists(expectedCache));
    }

    SECTION("windows returns UTF-8 paths for non-ASCII app data") {
        TempPathRoot temp("windows-unicode");
        const auto appDataRoot = temp.root / std::filesystem::path(L"Roaming-\u00dcnicode");
        const auto localAppDataRoot = temp.root / std::filesystem::path(L"Local-\u65e5\u672c\u8a9e");
        const auto expectedConfig = appDataRoot / "CyberEther";
        const auto expectedCache = localAppDataRoot / "CyberEther" / "Cache";

        const ScopedWideEnvVar appDataEnv(L"APPDATA");
        const ScopedWideEnvVar localAppDataEnv(L"LOCALAPPDATA");

        REQUIRE(appDataEnv.set(appDataRoot.native()));
        REQUIRE(localAppDataEnv.set(localAppDataRoot.native()));

        std::string configPath;
        std::string cachePath;
        SeedPaths(configPath, cachePath);
        REQUIRE(Platform::ConfigPath(configPath) == Result::SUCCESS);
        REQUIRE(Platform::CachePath(cachePath) == Result::SUCCESS);

        REQUIRE(configPath == PathToUtf8(expectedConfig));
        REQUIRE(cachePath == PathToUtf8(expectedCache));
        REQUIRE(!std::filesystem::exists(expectedConfig));
        REQUIRE(!std::filesystem::exists(expectedCache));
    }

    SECTION("windows missing APPDATA fails config resolution") {
        const ScopedEnvVar appDataEnv("APPDATA");

        REQUIRE(appDataEnv.set(std::nullopt));

        std::string configPath;
        std::string cachePath;
        SeedPaths(configPath, cachePath);
        REQUIRE(Platform::ConfigPath(configPath) == Result::ERROR);
        REQUIRE(configPath == kInitialConfigPath);
        REQUIRE(cachePath == kInitialCachePath);
    }

    SECTION("windows missing app data fails cache resolution") {
        const ScopedEnvVar appDataEnv("APPDATA");
        const ScopedEnvVar localAppDataEnv("LOCALAPPDATA");

        REQUIRE(appDataEnv.set(std::nullopt));
        REQUIRE(localAppDataEnv.set(std::nullopt));

        std::string configPath;
        std::string cachePath;
        SeedPaths(configPath, cachePath);
        REQUIRE(Platform::CachePath(cachePath) == Result::ERROR);
        REQUIRE(configPath == kInitialConfigPath);
        REQUIRE(cachePath == kInitialCachePath);
    }

    SECTION("empty windows env values are treated as unavailable") {
        const ScopedEnvVar appDataEnv("APPDATA");
        const ScopedEnvVar localAppDataEnv("LOCALAPPDATA");

        REQUIRE(appDataEnv.set(std::string()));
        REQUIRE(localAppDataEnv.set(std::string()));

        std::string configPath;
        std::string cachePath;
        SeedPaths(configPath, cachePath);
        REQUIRE(Platform::ConfigPath(configPath) == Result::ERROR);
        REQUIRE(Platform::CachePath(cachePath) == Result::ERROR);
        REQUIRE(configPath == kInitialConfigPath);
        REQUIRE(cachePath == kInitialCachePath);
    }
#elif defined(JST_OS_MAC) || defined(JST_OS_IOS)
    SECTION("apple resolves app-specific config and cache directories") {
        std::string configPath;
        std::string cachePath;
        SeedPaths(configPath, cachePath);
        REQUIRE(Platform::ConfigPath(configPath) == Result::SUCCESS);
        REQUIRE(Platform::CachePath(cachePath) == Result::SUCCESS);

        const std::filesystem::path config(configPath);
        const std::filesystem::path cache(cachePath);
        REQUIRE(config.is_absolute());
        REQUIRE(cache.is_absolute());
        REQUIRE(config.filename() == "CyberEther");
        REQUIRE(cache.filename() == "CyberEther");
        REQUIRE(config.parent_path().filename() == "Application Support");
        REQUIRE(cache.parent_path().filename() == "Caches");
        REQUIRE(config != cache);
    }
#elif defined(JST_OS_BROWSER)
    SECTION("browser returns stable virtual storage paths") {
        std::string configPath;
        std::string cachePath;
        SeedPaths(configPath, cachePath);
        REQUIRE(Platform::ConfigPath(configPath) == Result::SUCCESS);
        REQUIRE(Platform::CachePath(cachePath) == Result::SUCCESS);

        REQUIRE(configPath == "/storage/cyberether");
        REQUIRE(cachePath == "/storage/cyberether/cache");
    }
#else
    SECTION("unsupported platforms report errors") {
        std::string configPath = "set";
        std::string cachePath = "set";
        REQUIRE(Platform::ConfigPath(configPath) == Result::ERROR);
        REQUIRE(Platform::CachePath(cachePath) == Result::ERROR);
        REQUIRE(configPath.empty());
        REQUIRE(cachePath.empty());
    }
#endif
}
