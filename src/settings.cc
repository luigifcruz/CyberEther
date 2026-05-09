#include "jetstream/settings.hh"

#include <filesystem>
#include <fstream>
#include <mutex>

#include "jetstream/platform.hh"

namespace Jetstream {

struct Settings::Impl {
    static constexpr const char* Filename = "settings.yaml";

    static Impl& Instance();
    static Result ResolvePath(std::filesystem::path& path);
    static std::filesystem::path ResolveTempPath(const std::filesystem::path& path);
    static Result LoadFile(const std::filesystem::path& path, Settings& settings);
    static Result SaveFile(const std::filesystem::path& path, const Settings& settings);

    std::mutex mutex;
    bool loaded = false;
    std::filesystem::path path;
    Settings settings;
};

Settings::Impl& Settings::Impl::Instance() {
    static Impl impl;
    return impl;
}

Result Settings::Impl::ResolvePath(std::filesystem::path& path) {
    std::string configPath;
    JST_CHECK(Platform::ConfigPath(configPath));

    path = std::filesystem::u8path(configPath) / Filename;
    return Result::SUCCESS;
}

std::filesystem::path Settings::Impl::ResolveTempPath(const std::filesystem::path& path) {
    auto tempPath = path;
    tempPath += ".tmp";
    return tempPath;
}

Result Settings::Impl::LoadFile(const std::filesystem::path& path, Settings& settings) {
    settings = {};

    std::error_code ec;
    const bool exists = std::filesystem::exists(path, ec);
    if (ec) {
        JST_ERROR("[SETTINGS] Failed to query settings file '{}'.", path.string());
        return Result::ERROR;
    }

    if (!exists) {
        return Result::SUCCESS;
    }

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        JST_ERROR("[SETTINGS] Can't open settings file '{}'.", path.string());
        return Result::ERROR;
    }

    const std::string yaml((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    Parser::Map data;
    JST_CHECK(Parser::YamlDecode(yaml, data));
    return settings.deserialize(data);
}

Result Settings::Impl::SaveFile(const std::filesystem::path& path, const Settings& settings) {
    const auto parent = path.parent_path();
    if (!parent.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
        if (ec) {
            JST_ERROR("[SETTINGS] Cannot create directory '{}'.", parent.string());
            return Result::ERROR;
        }
    }

    Parser::Map data;
    JST_CHECK(settings.serialize(data));

    std::string yaml;
    JST_CHECK(Parser::YamlEncode(data, yaml));

    const auto tempPath = ResolveTempPath(path);

    std::ofstream file(tempPath, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!file) {
        JST_ERROR("[SETTINGS] Can't open temporary settings file '{}'.", tempPath.string());
        return Result::ERROR;
    }

    file.write(yaml.data(), static_cast<std::streamsize>(yaml.size()));
    if (!file) {
        file.close();

        std::error_code cleanupEc;
        (void)std::filesystem::remove(tempPath, cleanupEc);

        JST_ERROR("[SETTINGS] Failed to write temporary settings file '{}'.", tempPath.string());
        return Result::ERROR;
    }

    file.close();
    if (!file) {
        std::error_code cleanupEc;
        (void)std::filesystem::remove(tempPath, cleanupEc);

        JST_ERROR("[SETTINGS] Failed to finalize temporary settings file '{}'.", tempPath.string());
        return Result::ERROR;
    }

    std::error_code ec;
    std::filesystem::rename(tempPath, path, ec);
    if (ec) {
        std::error_code cleanupEc;
        (void)std::filesystem::remove(tempPath, cleanupEc);

        JST_ERROR("[SETTINGS] Failed to replace settings file '{}'.", path.string());
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result Settings::Get(Settings& settings) {
    std::filesystem::path path;
    JST_CHECK(Impl::ResolvePath(path));

    Impl& impl = Impl::Instance();
    std::lock_guard lock(impl.mutex);
    if (!impl.loaded || impl.path != path) {
        JST_CHECK(Impl::LoadFile(path, impl.settings));
        impl.loaded = true;
        impl.path = path;
    }

    settings = impl.settings;
    return Result::SUCCESS;
}

Result Settings::Set(const Settings& settings, bool persist) {
    std::filesystem::path path;
    JST_CHECK(Impl::ResolvePath(path));

    Impl& impl = Impl::Instance();
    std::lock_guard lock(impl.mutex);
    impl.settings = settings;
    impl.loaded = true;
    impl.path = path;

    if (persist) {
        JST_CHECK(Impl::SaveFile(path, impl.settings));
    }

    return Result::SUCCESS;
}

}  // namespace Jetstream
