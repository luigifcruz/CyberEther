#include "jetstream/platform.hh"

#include <cstdlib>
#include <filesystem>
#include <utility>

#if defined(JST_OS_WINDOWS)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#undef ERROR
#undef FATAL
#endif

namespace Jetstream::Platform {

std::filesystem::path PathFromUtf8(const std::string& path) {
    return std::filesystem::path(std::u8string(path.begin(), path.end()));
}

std::string PathToUtf8(const std::filesystem::path& path) {
    const auto utf8Path = path.u8string();
    return std::string(utf8Path.begin(), utf8Path.end());
}

Result EnvironmentPath(const std::string& name, std::filesystem::path& path) {
    try {
#if defined(JST_OS_WINDOWS)
        const auto nativeName = PathFromUtf8(name).native();
        const DWORD requiredSize = GetEnvironmentVariableW(nativeName.c_str(), nullptr, 0);
        if (requiredSize == 0) {
            return Result::ERROR;
        }

        std::wstring value(requiredSize, L'\0');
        const DWORD writtenSize = GetEnvironmentVariableW(
            nativeName.c_str(), value.data(), requiredSize);
        if (writtenSize == 0 || writtenSize >= requiredSize) {
            return Result::ERROR;
        }

        value.resize(writtenSize);
        path = std::filesystem::path(std::move(value));
#else
        const char* value = std::getenv(name.c_str());
        if (!value || !*value) {
            return Result::ERROR;
        }

        path = std::filesystem::path(value);
#endif
    } catch (...) {
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

#if defined(JST_OS_MAC) || defined(JST_OS_IOS)

// Defined on apple.mm.

#elif defined(JST_OS_BROWSER)

Result ConfigPath(std::string& path) {
    path = "/storage/cyberether";
    return Result::SUCCESS;
}

Result CachePath(std::string& path) {
    path = "/storage/cyberether/cache";
    return Result::SUCCESS;
}

#elif defined(JST_OS_LINUX)

Result ConfigPath(std::string& path) {
    std::filesystem::path basePath;
    if (EnvironmentPath("XDG_CONFIG_HOME", basePath) == Result::SUCCESS) {
        if (basePath.is_absolute()) {
            path = PathToUtf8(basePath / "cyberether");
            return Result::SUCCESS;
        }

        JST_WARN("XDG_CONFIG_HOME must be an absolute path. Falling back to $HOME/.config.");
    }

    std::filesystem::path homePath;
    if (EnvironmentPath("HOME", homePath) != Result::SUCCESS) {
        JST_ERROR("Failed to resolve app path because HOME is not set.");
        return Result::ERROR;
    }
    path = PathToUtf8(homePath / ".config" / "cyberether");
    return Result::SUCCESS;
}

Result CachePath(std::string& path) {
    std::filesystem::path basePath;
    if (EnvironmentPath("XDG_CACHE_HOME", basePath) == Result::SUCCESS) {
        if (basePath.is_absolute()) {
            path = PathToUtf8(basePath / "cyberether");
            return Result::SUCCESS;
        }

        JST_WARN("XDG_CACHE_HOME must be an absolute path. Falling back to $HOME/.cache.");
    }

    std::filesystem::path homePath;
    if (EnvironmentPath("HOME", homePath) != Result::SUCCESS) {
        JST_ERROR("Failed to resolve app path because HOME is not set.");
        return Result::ERROR;
    }
    path = PathToUtf8(homePath / ".cache" / "cyberether");
    return Result::SUCCESS;
}

#elif defined(JST_OS_WINDOWS)

Result ConfigPath(std::string& path) {
    std::filesystem::path appData;
    if (EnvironmentPath("APPDATA", appData) != Result::SUCCESS) {
        JST_ERROR("Failed to resolve config path because APPDATA is not set.");
        return Result::ERROR;
    }

    path = PathToUtf8(appData / L"CyberEther");
    return Result::SUCCESS;
}

Result CachePath(std::string& path) {
    std::filesystem::path appData;
    std::filesystem::path localAppData;

    if (EnvironmentPath("LOCALAPPDATA", localAppData) == Result::SUCCESS) {
        path = PathToUtf8(localAppData / L"CyberEther" / L"Cache");
        return Result::SUCCESS;
    }

    if (EnvironmentPath("APPDATA", appData) == Result::SUCCESS) {
        path = PathToUtf8(appData / L"CyberEther" / L"Cache");
        return Result::SUCCESS;
    }

    JST_ERROR("Failed to resolve cache path because LOCALAPPDATA and APPDATA are not set.");
    return Result::ERROR;
}

#else

Result ConfigPath(std::string& path) {
    path.clear();
    JST_ERROR("Resolving configuration paths is not supported on this platform.");
    return Result::ERROR;
}

Result CachePath(std::string& path) {
    path.clear();
    JST_ERROR("Resolving cache paths is not supported on this platform.");
    return Result::ERROR;
}

#endif

}  // namespace Jetstream::Platform
