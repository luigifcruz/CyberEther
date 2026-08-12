#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "jetstream/platform.hh"

#if defined(JST_OS_WINDOWS)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#undef ERROR
#undef FATAL
#elif defined(JST_OS_LINUX) || defined(JST_OS_MAC)
#include <dlfcn.h>
#endif

using namespace Jetstream;

namespace {

constexpr const char* kInitialConfigPath = "existing-config";
constexpr const char* kInitialCachePath = "existing-cache";
#if defined(JST_OS_WINDOWS) || defined(JST_OS_LINUX) || defined(JST_OS_MAC)
constexpr std::size_t kProcessOutputLimit = 1024 * 1024;
#endif

void SeedPaths(std::string& configPath, std::string& cachePath) {
    configPath = kInitialConfigPath;
    cachePath = kInitialCachePath;
}

bool SetEnvValue(const char* name, const std::optional<std::string>& value) {
#if defined(JST_OS_WINDOWS)
    try {
        const auto nativeName = Platform::PathFromUtf8(name).native();
        const auto nativeValue = value ? Platform::PathFromUtf8(*value).native() : std::wstring();
        return SetEnvironmentVariableW(nativeName.c_str(), value ? nativeValue.c_str() : nullptr) !=
               FALSE;
    } catch (...) {
        return false;
    }
#else
    if (value) {
        return setenv(name, value->c_str(), 1) == 0;
    }

    return unsetenv(name) == 0;
#endif
}

#if defined(JST_OS_WINDOWS)

bool SetWideEnvValue(const wchar_t* name, const std::optional<std::wstring>& value) {
    return SetEnvironmentVariableW(name, value ? value->c_str() : nullptr) != FALSE;
}

struct ScopedWideEnvVar {
    explicit ScopedWideEnvVar(const wchar_t* name) : name(name) {
        std::string value;
        const auto utf8Name = Platform::PathToUtf8(std::filesystem::path(name));
        if (Platform::EnvironmentVariable(utf8Name, value) == Result::SUCCESS) {
            originalValue = Platform::PathFromUtf8(value).native();
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
        std::string value;
        if (Platform::EnvironmentVariable(name, value) == Result::SUCCESS) {
            originalValue = std::move(value);
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
        const auto parent = std::filesystem::temp_directory_path();
        for (std::size_t attempt = 0; attempt < 1024; ++attempt) {
            const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
            root = parent / Platform::PathFromUtf8(
                                "cyberether-platform-" + label + "-" +
                                std::to_string(nonce) + "-" + std::to_string(attempt));

            std::error_code ec;
            if (std::filesystem::create_directory(root, ec)) {
                return;
            }
            if (ec && ec != std::errc::file_exists) {
                throw std::runtime_error("failed to create platform test directory");
            }
        }

        throw std::runtime_error("failed to allocate unique platform test directory");
    }

    ~TempPathRoot() {
        std::error_code ec;
        std::filesystem::remove_all(root, ec);
    }

    TempPathRoot(const TempPathRoot&) = delete;
    TempPathRoot& operator=(const TempPathRoot&) = delete;

    std::filesystem::path root;
};

#if defined(JST_OS_WINDOWS) || defined(JST_OS_LINUX) || defined(JST_OS_MAC)

void WriteBinaryFile(const std::filesystem::path& path, std::string_view contents) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    REQUIRE(stream.is_open());
    stream.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    stream.close();
    REQUIRE_FALSE(stream.fail());
}

std::string ReadBinaryFile(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    REQUIRE(stream.is_open());
    std::string contents{
        std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
    REQUIRE_FALSE(stream.bad());
    return contents;
}

Result RunFilePrinter(const std::filesystem::path& path,
                      std::string& output,
                      U64 timeoutMilliseconds = 5000) {
#if defined(JST_OS_WINDOWS)
    return Platform::RunProcess(
        "cmd.exe", {"/D", "/C", "type", Platform::PathToUtf8(path)}, output, timeoutMilliseconds);
#else
    return Platform::RunProcess(
        "/bin/cat", {Platform::PathToUtf8(path)}, output, timeoutMilliseconds);
#endif
}

#endif

struct ScopedDynamicLibrary {
    ~ScopedDynamicLibrary() {
        Platform::CloseDynamicLibrary(handle);
    }

    void* handle = nullptr;
};

}  // namespace

TEST_CASE("Platform paths preserve UTF-8", "[core][platform][paths]") {
    REQUIRE(Platform::PathFromUtf8("").empty());
    REQUIRE(Platform::PathToUtf8({}).empty());

    const std::string utf8Path = "CyberEther-\xC3\x9C-\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E";
    REQUIRE(Platform::PathToUtf8(Platform::PathFromUtf8(utf8Path)) == utf8Path);
}

TEST_CASE("Platform paths preserve lexical boundaries", "[core][platform][paths]") {
    const std::string utf8Leaf = "leaf-\xC3\x9C-\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E";
    const auto parent = Platform::PathFromUtf8("parent with spaces");
    const auto leaf = Platform::PathFromUtf8(utf8Leaf);
    const auto relative = parent / leaf;

    REQUIRE(relative.is_relative());
    REQUIRE(relative.parent_path() == parent);
    REQUIRE(relative.filename() == leaf);
    REQUIRE(Platform::PathFromUtf8(Platform::PathToUtf8(relative)) == relative);

    const std::string longComponent(4096, 'x');
    REQUIRE(Platform::PathToUtf8(Platform::PathFromUtf8(longComponent)) == longComponent);

    TempPathRoot temp("path-boundaries");
    const auto absolute = temp.root / leaf;
    REQUIRE(absolute.is_absolute());
    REQUIRE(Platform::PathFromUtf8(Platform::PathToUtf8(absolute)) == absolute);
    REQUIRE(Platform::PathFromUtf8(Platform::PathToUtf8(absolute.root_path())) ==
            absolute.root_path());
}

TEST_CASE("Platform environment variables preserve values", "[core][platform][environment]") {
    const ScopedEnvVar environment("CYBERETHER_TEST_ENVIRONMENT_VARIABLE");
    REQUIRE(environment.set("cyberether environment=value"));

    std::string value;
    REQUIRE(Platform::EnvironmentVariable(environment.name, value) == Result::SUCCESS);
    REQUIRE(value == "cyberether environment=value");

#if !defined(JST_OS_WINDOWS)
    REQUIRE(environment.set(std::string()));
    value = "unchanged";
    REQUIRE(Platform::EnvironmentVariable(environment.name, value) == Result::SUCCESS);
    REQUIRE(value.empty());
#endif

    REQUIRE(environment.set(std::nullopt));
    value = "unchanged";
    REQUIRE(Platform::EnvironmentVariable(environment.name, value) == Result::ERROR);
    REQUIRE(value == "unchanged");
}

TEST_CASE("Platform environment fixtures restore present and absent values",
          "[core][platform][environment]") {
    const ScopedEnvVar externalEnvironment("CYBERETHER_TEST_ENVIRONMENT_RESTORATION");

    REQUIRE(externalEnvironment.set("baseline value"));
    {
        const ScopedEnvVar restorePresent(externalEnvironment.name);
        REQUIRE(restorePresent.set("temporary value"));

        std::string value;
        REQUIRE(Platform::EnvironmentVariable(externalEnvironment.name, value) == Result::SUCCESS);
        REQUIRE(value == "temporary value");
    }

    std::string value;
    REQUIRE(Platform::EnvironmentVariable(externalEnvironment.name, value) == Result::SUCCESS);
    REQUIRE(value == "baseline value");

    REQUIRE(externalEnvironment.set(std::nullopt));
    {
        const ScopedEnvVar restoreAbsent(externalEnvironment.name);
        REQUIRE(restoreAbsent.set("temporary value"));
    }

    value = "unchanged";
    REQUIRE(Platform::EnvironmentVariable(externalEnvironment.name, value) == Result::ERROR);
    REQUIRE(value == "unchanged");
}

TEST_CASE("Platform environment lookups reject empty names", "[core][platform][environment]") {
    std::string value = "unchanged";
    REQUIRE(Platform::EnvironmentVariable("", value) == Result::ERROR);
    REQUIRE(value == "unchanged");

    std::filesystem::path path = "unchanged";
    REQUIRE(Platform::EnvironmentPath("", path) == Result::ERROR);
    REQUIRE(path == "unchanged");
}

TEST_CASE("Platform environment paths are native", "[core][platform][environment]") {
    const ScopedEnvVar environment("CYBERETHER_TEST_ENVIRONMENT_PATH");
    const std::string utf8Path =
        "cyberether-\xC3\x9C-\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E/path";
    REQUIRE(environment.set(utf8Path));

    std::filesystem::path path;
    REQUIRE(Platform::EnvironmentPath(environment.name, path) == Result::SUCCESS);
    REQUIRE(path == Platform::PathFromUtf8(utf8Path));

    REQUIRE(environment.set(std::string()));
    path = "unchanged";
    REQUIRE(Platform::EnvironmentPath(environment.name, path) == Result::ERROR);
    REQUIRE(path == "unchanged");

    REQUIRE(environment.set(std::nullopt));
    path = "unchanged";
    REQUIRE(Platform::EnvironmentPath(environment.name, path) == Result::ERROR);
    REQUIRE(path == "unchanged");

#if defined(JST_OS_WINDOWS)
    const ScopedWideEnvVar wideEnvironment(L"CYBERETHER_TEST_WIDE_ENVIRONMENT_PATH");
    const std::wstring widePath = L"cyberether-\u00dcnicode-\u65e5\u672c\u8a9e";
    REQUIRE(wideEnvironment.set(widePath));
    REQUIRE(Platform::EnvironmentPath("CYBERETHER_TEST_WIDE_ENVIRONMENT_PATH", path) ==
            Result::SUCCESS);
    REQUIRE(path == std::filesystem::path(widePath));
#endif
}

#if defined(JST_OS_WINDOWS) || defined(JST_OS_LINUX) || defined(JST_OS_MAC)

TEST_CASE("Platform file locks support lifecycle and reacquisition",
          "[core][platform][file-lock]") {
    TempPathRoot temp("file-lock-lifecycle");
    const auto lockPath = temp.root / "state.lock";
    const auto lockPathUtf8 = Platform::PathToUtf8(lockPath);
    const std::string contents = "existing lock contents\n";
    WriteBinaryFile(lockPath, contents);

    Platform::FileLock lock;
    REQUIRE_FALSE(lock.locked());
    REQUIRE(lock.acquire(lockPathUtf8, false) == Result::SUCCESS);
    REQUIRE(lock.locked());
    REQUIRE(std::filesystem::file_size(lockPath) == contents.size());

    REQUIRE(lock.acquire(lockPathUtf8, false) == Result::ERROR);
    REQUIRE(lock.locked());

    lock.release();
    REQUIRE_FALSE(lock.locked());
    lock.release();
    REQUIRE_FALSE(lock.locked());

    REQUIRE(lock.acquire(lockPathUtf8, false) == Result::SUCCESS);
    REQUIRE(lock.locked());
    lock.release();
    REQUIRE_FALSE(lock.locked());
    REQUIRE(ReadBinaryFile(lockPath) == contents);
}

TEST_CASE("Platform file locks report contention and release on destruction",
          "[core][platform][file-lock]") {
    TempPathRoot temp("file-lock-contention");
    const auto lockPath = Platform::PathToUtf8(temp.root / "contention.lock");

    Platform::FileLock owner;
    REQUIRE(owner.acquire(lockPath, false) == Result::SUCCESS);

    {
        Platform::FileLock contender;
        REQUIRE(contender.acquire(lockPath, false) == Result::SKIP);
        REQUIRE_FALSE(contender.locked());

        owner.release();
        REQUIRE(contender.acquire(lockPath, false) == Result::SUCCESS);
        REQUIRE(contender.locked());
        REQUIRE(owner.acquire(lockPath, false) == Result::SKIP);
        REQUIRE_FALSE(owner.locked());
    }

    REQUIRE(owner.acquire(lockPath, false) == Result::SUCCESS);
    REQUIRE(owner.locked());
}

TEST_CASE("Platform file locks transfer ownership through moves",
          "[core][platform][file-lock]") {
    TempPathRoot temp("file-lock-move");
    const auto primaryPath = Platform::PathToUtf8(temp.root / "primary.lock");
    const auto replacementPath = Platform::PathToUtf8(temp.root / "replacement.lock");

    Platform::FileLock original;
    REQUIRE(original.acquire(primaryPath, false) == Result::SUCCESS);

    Platform::FileLock moved(std::move(original));
    REQUIRE_FALSE(original.locked());
    REQUIRE(moved.locked());

    REQUIRE(original.acquire(replacementPath, false) == Result::SUCCESS);
    original = std::move(moved);
    REQUIRE(original.locked());
    REQUIRE_FALSE(moved.locked());

    Platform::FileLock releasedReplacement;
    REQUIRE(releasedReplacement.acquire(replacementPath, false) == Result::SUCCESS);

    Platform::FileLock primaryContender;
    REQUIRE(primaryContender.acquire(primaryPath, false) == Result::SKIP);
    original.release();
    REQUIRE(primaryContender.acquire(primaryPath, false) == Result::SUCCESS);

    REQUIRE(moved.acquire(Platform::PathToUtf8(temp.root / "moved-from.lock"), false) ==
            Result::SUCCESS);
    REQUIRE(moved.locked());
}

TEST_CASE("Platform file locks reject invalid paths", "[core][platform][file-lock]") {
    TempPathRoot temp("file-lock-invalid");
    Platform::FileLock lock;

    SECTION("empty path") {
        REQUIRE(lock.acquire("", false) == Result::ERROR);
        REQUIRE_FALSE(lock.locked());
    }

    SECTION("parent component is a file") {
        const auto parentFile = temp.root / "not-a-directory";
        WriteBinaryFile(parentFile, "contents");

        REQUIRE(lock.acquire(Platform::PathToUtf8(parentFile / "lock"), false) == Result::ERROR);
        REQUIRE_FALSE(lock.locked());
        REQUIRE(ReadBinaryFile(parentFile) == "contents");
    }

    SECTION("path is an existing directory") {
        REQUIRE(lock.acquire(Platform::PathToUtf8(temp.root), false) == Result::ERROR);
        REQUIRE_FALSE(lock.locked());
    }
}

#endif

TEST_CASE("Platform dynamic library errors include a reason",
           "[core][platform][dynamic-library]") {
    TempPathRoot temp("missing-library");
    const auto missingLibrary = Platform::PathToUtf8(temp.root / "missing-library");
    REQUIRE(!std::filesystem::exists(temp.root / "missing-library"));

    const auto requireOpenFailure = [&](Platform::DynamicLibraryVisibility visibility) {
        std::string error = "stale error";
        void* handle = Platform::OpenDynamicLibrary(missingLibrary, visibility, error);

        REQUIRE(handle == nullptr);
        REQUIRE(!error.empty());
        REQUIRE(error != "stale error");
        Platform::CloseDynamicLibrary(handle);
    };

    SECTION("local visibility") {
        requireOpenFailure(Platform::DynamicLibraryVisibility::Local);
    }

    SECTION("global visibility") {
        requireOpenFailure(Platform::DynamicLibraryVisibility::Global);
    }
}

#if defined(JST_OS_WINDOWS) || defined(JST_OS_LINUX) || defined(JST_OS_MAC)

TEST_CASE("Platform dynamic library wrapper opens a loaded system library",
          "[core][platform][dynamic-library]") {
    ScopedDynamicLibrary library;
    std::string libraryPath;
    const char* symbolName = nullptr;

#if defined(JST_OS_WINDOWS)
    libraryPath = "kernel32.dll";
    symbolName = "GetCurrentProcessId";
#elif defined(JST_OS_LINUX) || defined(JST_OS_MAC)
    Dl_info libraryInfo = {};
    const auto loaderAddress = reinterpret_cast<std::uintptr_t>(&dlopen);
    REQUIRE(dladdr(reinterpret_cast<const void*>(loaderAddress), &libraryInfo) != 0);
    REQUIRE(libraryInfo.dli_fname != nullptr);
    REQUIRE(*libraryInfo.dli_fname != '\0');
    libraryPath = libraryInfo.dli_fname;
    symbolName = "dlopen";
#endif

    std::string error = "stale error";
    library.handle = Platform::OpenDynamicLibrary(
        libraryPath, Platform::DynamicLibraryVisibility::Local, error);
    REQUIRE(library.handle != nullptr);
    REQUIRE(error.empty());

    error = "stale error";
    REQUIRE(Platform::LoadDynamicLibrarySymbol(library.handle, symbolName, error) != nullptr);
    REQUIRE(error.empty());
}

TEST_CASE("Platform dynamic library symbol errors include a reason",
           "[core][platform][dynamic-library]") {
    ScopedDynamicLibrary library;
    std::string error;

#if defined(JST_OS_WINDOWS)
    library.handle = Platform::OpenDynamicLibrary(
        "kernel32.dll", Platform::DynamicLibraryVisibility::Local, error);
    REQUIRE(library.handle != nullptr);
    REQUIRE(error.empty());
#elif defined(JST_OS_LINUX) || defined(JST_OS_MAC)
    library.handle = dlopen(nullptr, RTLD_NOW | RTLD_LOCAL);
    REQUIRE(library.handle != nullptr);
#endif

    error = "stale error";
    void* symbol = Platform::LoadDynamicLibrarySymbol(
        library.handle, "cyberether_symbol_that_does_not_exist_8f59f725", error);
    REQUIRE(symbol == nullptr);
    REQUIRE(!error.empty());
    REQUIRE(error != "stale error");
}

#endif

TEST_CASE("Platform processes preserve output transactionally", "[core][platform][process]") {
    std::string output = "unchanged";
    REQUIRE(Platform::RunProcess("", {}, output, 5000) == Result::ERROR);
    REQUIRE(output == "unchanged");

#if defined(JST_OS_WINDOWS)
    REQUIRE(Platform::RunProcess("cmd.exe", {"/D", "/C", "echo cyberether-process"}, output, 5000) ==
            Result::SUCCESS);
    REQUIRE(output == "cyberether-process\r\n");
#elif defined(JST_OS_LINUX) || defined(JST_OS_MAC)
    REQUIRE(Platform::RunProcess("/bin/echo", {"cyberether-process"}, output, 5000) ==
            Result::SUCCESS);
    REQUIRE(output == "cyberether-process\n");
#else
    REQUIRE(Platform::RunProcess("unsupported", {}, output, 5000) == Result::ERROR);
    REQUIRE(output == "unchanged");
    return;
#endif

    output = "unchanged";
#if defined(JST_OS_WINDOWS)
    REQUIRE(Platform::RunProcess("cmd.exe", {"/D", "/C", "exit /B 0"}, output, 5000) ==
            Result::SUCCESS);
#else
    REQUIRE(Platform::RunProcess("/bin/sh", {"-c", "exit 0"}, output, 5000) == Result::SUCCESS);
#endif
    REQUIRE(output.empty());

    output = "unchanged";
#if defined(JST_OS_WINDOWS)
    REQUIRE(Platform::RunProcess(
                "cmd.exe", {"/D", "/C", "echo ignored & exit /B 7"}, output, 5000) ==
            Result::ERROR);
#else
    REQUIRE(Platform::RunProcess(
                "/bin/sh", {"-c", "printf ignored; exit 7"}, output, 5000) == Result::ERROR);
#endif
    REQUIRE(output == "unchanged");

    output = "unchanged";
    REQUIRE(Platform::RunProcess("cyberether-process-that-does-not-exist", {}, output, 5000) ==
            Result::ERROR);
    REQUIRE(output == "unchanged");
}

#if defined(JST_OS_WINDOWS) || defined(JST_OS_LINUX) || defined(JST_OS_MAC)

TEST_CASE("Platform processes preserve arguments and binary output",
          "[core][platform][process]") {
#if defined(JST_OS_LINUX) || defined(JST_OS_MAC)
    std::string argumentOutput = "unchanged";
    REQUIRE(Platform::RunProcess(
                "/bin/sh",
                {"-c",
                 "for argument do printf '<%s>\\n' \"$argument\"; done",
                 "cyberether-argv-zero",
                 "",
                 "plain",
                 "two words",
                 "single'quote",
                 "double\"quote",
                 "trailing\\",
                 "utf8-\xC3\x9C-\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E"},
                argumentOutput,
                0) == Result::SUCCESS);
    REQUIRE(argumentOutput ==
            "<>\n"
            "<plain>\n"
            "<two words>\n"
            "<single'quote>\n"
            "<double\"quote>\n"
            "<trailing\\>\n"
            "<utf8-\xC3\x9C-\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E>\n");
#endif

    TempPathRoot temp("process-binary");
    const auto binaryPath =
        temp.root /
        Platform::PathFromUtf8(
            "binary output-\xC3\x9C-\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E.bin");
    std::string binaryOutput = "binary";
    binaryOutput.push_back('\0');
    binaryOutput.append(
        "output-\xC3\x9C-\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E");
    binaryOutput.push_back('\x01');
    binaryOutput.push_back('\x7f');
    WriteBinaryFile(binaryPath, binaryOutput);

    std::string output = "unchanged";
    REQUIRE(RunFilePrinter(binaryPath, output, 0) == Result::SUCCESS);
    REQUIRE(output == binaryOutput);
}

TEST_CASE("Platform processes capture stdout without stderr",
          "[core][platform][process]") {
    std::string output = "unchanged";
#if defined(JST_OS_WINDOWS)
    REQUIRE(Platform::RunProcess(
                "cmd.exe", {"/D", "/C", "echo stdout& echo stderr 1>&2"}, output, 5000) ==
            Result::SUCCESS);
    REQUIRE(output == "stdout\r\n");
#else
    REQUIRE(Platform::RunProcess(
                "/bin/sh",
                {"-c", "printf stdout; printf stderr >&2"},
                output,
                5000) == Result::SUCCESS);
    REQUIRE(output == "stdout");
#endif
}

TEST_CASE("Platform process timeouts leave output transactional",
          "[core][platform][process]") {
    std::string output = "unchanged";
#if defined(JST_OS_WINDOWS)
    REQUIRE(Platform::RunProcess(
                "cmd.exe",
                {"/D",
                 "/C",
                 "echo partial & for /L %i in (1,1,1000000) do @rem finite-delay"},
                output,
                1) == Result::ERROR);
#else
    REQUIRE(Platform::RunProcess(
                "/bin/sh",
                {"-c",
                 "printf partial; i=0; while [ \"$i\" -lt 1000000 ]; do i=$((i+1)); done"},
                output,
                1) == Result::ERROR);
#endif
    REQUIRE(output == "unchanged");
}

TEST_CASE("Platform process output limit is exact and transactional",
          "[core][platform][process]") {
    TempPathRoot temp("process-output-limit");
    std::string output = "unchanged";

    SECTION("exact limit succeeds") {
        const std::string expected(kProcessOutputLimit, 'x');
        const auto path = temp.root / "exact output limit.bin";
        WriteBinaryFile(path, expected);

        REQUIRE(RunFilePrinter(path, output, 10000) == Result::SUCCESS);
        REQUIRE(output == expected);
    }

    SECTION("one byte over the limit fails transactionally") {
        const std::string oversized(kProcessOutputLimit + 1, 'x');
        const auto path = temp.root / "oversized output.bin";
        WriteBinaryFile(path, oversized);

        REQUIRE(RunFilePrinter(path, output, 10000) == Result::ERROR);
        REQUIRE(output == "unchanged");
    }
}

#endif

TEST_CASE("Platform config and cache paths follow platform conventions",
          "[core][platform][paths]") {
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
        REQUIRE(xdgConfigEnv.set(Platform::PathToUtf8(configRoot)));
        REQUIRE(xdgCacheEnv.set(Platform::PathToUtf8(cacheRoot)));

        std::string configPath;
        std::string cachePath;
        SeedPaths(configPath, cachePath);
        REQUIRE(Platform::ConfigPath(configPath) == Result::SUCCESS);
        REQUIRE(Platform::CachePath(cachePath) == Result::SUCCESS);

        REQUIRE(Platform::PathFromUtf8(configPath) == expectedConfig);
        REQUIRE(Platform::PathFromUtf8(cachePath) == expectedCache);
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

        REQUIRE(homeEnv.set(Platform::PathToUtf8(homeRoot)));
        REQUIRE(xdgConfigEnv.set(std::nullopt));
        REQUIRE(xdgCacheEnv.set(std::nullopt));

        std::string configPath;
        std::string cachePath;
        SeedPaths(configPath, cachePath);
        REQUIRE(Platform::ConfigPath(configPath) == Result::SUCCESS);
        REQUIRE(Platform::CachePath(cachePath) == Result::SUCCESS);

        REQUIRE(Platform::PathFromUtf8(configPath) == expectedConfig);
        REQUIRE(Platform::PathFromUtf8(cachePath) == expectedCache);
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

        REQUIRE(homeEnv.set(Platform::PathToUtf8(homeRoot)));
        REQUIRE(xdgConfigEnv.set("relative-config"));
        REQUIRE(xdgCacheEnv.set("relative-cache"));

        std::string configPath;
        std::string cachePath;
        SeedPaths(configPath, cachePath);
        REQUIRE(Platform::ConfigPath(configPath) == Result::SUCCESS);
        REQUIRE(Platform::CachePath(cachePath) == Result::SUCCESS);

        REQUIRE(Platform::PathFromUtf8(configPath) == expectedConfig);
        REQUIRE(Platform::PathFromUtf8(cachePath) == expectedCache);
        REQUIRE(!std::filesystem::exists(homeRoot / ".config"));
        REQUIRE(!std::filesystem::exists(expectedConfig));
        REQUIRE(!std::filesystem::exists(homeRoot / ".cache"));
        REQUIRE(!std::filesystem::exists(expectedCache));
    }

    SECTION("relative HOME fails cleanly when XDG roots are unavailable") {
        const ScopedEnvVar homeEnv("HOME");
        const ScopedEnvVar xdgConfigEnv("XDG_CONFIG_HOME");
        const ScopedEnvVar xdgCacheEnv("XDG_CACHE_HOME");

        REQUIRE(homeEnv.set("relative-home"));
        REQUIRE(xdgConfigEnv.set(std::nullopt));
        REQUIRE(xdgCacheEnv.set(std::nullopt));

        std::string configPath;
        std::string cachePath;
        SeedPaths(configPath, cachePath);
        const auto configResult = Platform::ConfigPath(configPath);
        const auto cacheResult = Platform::CachePath(cachePath);

        // Current defect: Linux accepts relative HOME and returns relative application paths.
        CHECK(configResult == Result::ERROR);
        CHECK(cacheResult == Result::ERROR);
        CHECK(configPath == kInitialConfigPath);
        CHECK(cachePath == kInitialCachePath);
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

        REQUIRE(homeEnv.set(Platform::PathToUtf8(homeRoot)));
        REQUIRE(xdgConfigEnv.set(Platform::PathToUtf8(configRoot)));
        REQUIRE(xdgCacheEnv.set(std::nullopt));

        std::string configPath;
        std::string cachePath;
        SeedPaths(configPath, cachePath);
        REQUIRE(Platform::ConfigPath(configPath) == Result::SUCCESS);
        REQUIRE(Platform::CachePath(cachePath) == Result::SUCCESS);

        REQUIRE(Platform::PathFromUtf8(configPath) == expectedConfig);
        REQUIRE(Platform::PathFromUtf8(cachePath) == expectedCache);
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

        REQUIRE(homeEnv.set(Platform::PathToUtf8(homeRoot)));
        REQUIRE(xdgConfigEnv.set(std::nullopt));
        REQUIRE(xdgCacheEnv.set(Platform::PathToUtf8(cacheRoot)));

        std::string configPath;
        std::string cachePath;
        SeedPaths(configPath, cachePath);
        REQUIRE(Platform::ConfigPath(configPath) == Result::SUCCESS);
        REQUIRE(Platform::CachePath(cachePath) == Result::SUCCESS);

        REQUIRE(Platform::PathFromUtf8(configPath) == expectedConfig);
        REQUIRE(Platform::PathFromUtf8(cachePath) == expectedCache);
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

        REQUIRE(homeEnv.set(Platform::PathToUtf8(homeRoot)));
        REQUIRE(xdgConfigEnv.set(std::string()));
        REQUIRE(xdgCacheEnv.set(std::string()));

        std::string configPath;
        std::string cachePath;
        SeedPaths(configPath, cachePath);
        REQUIRE(Platform::ConfigPath(configPath) == Result::SUCCESS);
        REQUIRE(Platform::CachePath(cachePath) == Result::SUCCESS);

        REQUIRE(Platform::PathFromUtf8(configPath) == expectedConfig);
        REQUIRE(Platform::PathFromUtf8(cachePath) == expectedCache);
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

        REQUIRE(appDataEnv.set(Platform::PathToUtf8(appDataRoot)));
        REQUIRE(localAppDataEnv.set(Platform::PathToUtf8(localAppDataRoot)));

        std::string configPath;
        std::string cachePath;
        SeedPaths(configPath, cachePath);
        REQUIRE(Platform::ConfigPath(configPath) == Result::SUCCESS);
        REQUIRE(Platform::CachePath(cachePath) == Result::SUCCESS);

        REQUIRE(configPath == Platform::PathToUtf8(expectedConfig));
        REQUIRE(cachePath == Platform::PathToUtf8(expectedCache));
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

        REQUIRE(appDataEnv.set(Platform::PathToUtf8(appDataRoot)));
        REQUIRE(localAppDataEnv.set(std::nullopt));

        std::string configPath;
        std::string cachePath;
        SeedPaths(configPath, cachePath);
        REQUIRE(Platform::ConfigPath(configPath) == Result::SUCCESS);
        REQUIRE(Platform::CachePath(cachePath) == Result::SUCCESS);

        REQUIRE(configPath == Platform::PathToUtf8(expectedConfig));
        REQUIRE(cachePath == Platform::PathToUtf8(expectedCache));
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

        REQUIRE(configPath == Platform::PathToUtf8(expectedConfig));
        REQUIRE(cachePath == Platform::PathToUtf8(expectedCache));
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

    SECTION("relative windows app data roots fail cleanly") {
        const ScopedEnvVar appDataEnv("APPDATA");
        const ScopedEnvVar localAppDataEnv("LOCALAPPDATA");

        REQUIRE(appDataEnv.set("relative-roaming"));
        REQUIRE(localAppDataEnv.set("relative-local"));

        std::string configPath;
        std::string cachePath;
        SeedPaths(configPath, cachePath);
        const auto configResult = Platform::ConfigPath(configPath);
        const auto cacheResult = Platform::CachePath(cachePath);

        // Current defect: Windows accepts relative app-data roots and returns relative paths.
        CHECK(configResult == Result::ERROR);
        CHECK(cacheResult == Result::ERROR);
        CHECK(configPath == kInitialConfigPath);
        CHECK(cachePath == kInitialCachePath);
    }
#elif defined(JST_OS_MAC) || defined(JST_OS_IOS)
    SECTION("apple resolves app-specific config and cache directories") {
        std::string configPath;
        std::string cachePath;
        SeedPaths(configPath, cachePath);
        REQUIRE(Platform::ConfigPath(configPath) == Result::SUCCESS);
        REQUIRE(Platform::CachePath(cachePath) == Result::SUCCESS);

        const auto config = Platform::PathFromUtf8(configPath);
        const auto cache = Platform::PathFromUtf8(cachePath);
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
