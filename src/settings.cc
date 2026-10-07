#include "jetstream/settings.hh"

#include <filesystem>
#include <fstream>
#include <mutex>
#include <utility>

#include "jetstream/platform.hh"

namespace Jetstream {

struct Settings::Impl {
    static constexpr const char* Filename = "settings.yaml";

    static Impl& Instance();
    static Result ResolvePath(std::filesystem::path& path);
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

    path = Platform::PathFromUtf8(configPath) / Filename;
    return Result::SUCCESS;
}

Result Settings::Impl::LoadFile(const std::filesystem::path& path, Settings& settings) {
    settings = {};

    std::error_code ec;
    const bool exists = std::filesystem::exists(path, ec);
    if (ec) {
        JST_ERROR("[SETTINGS] Failed to query settings file '{}'.", Platform::PathToUtf8(path));
        return Result::ERROR;
    }

    if (!exists) {
        return Result::SUCCESS;
    }

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        JST_ERROR("[SETTINGS] Can't open settings file '{}'.", Platform::PathToUtf8(path));
        return Result::ERROR;
    }

    const std::string yaml((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    Parser::Map data;
    JST_CHECK(Parser::YamlDecode(yaml, data));
    return settings.deserialize(data);
}

Result Settings::Impl::SaveFile(const std::filesystem::path& path, const Settings& settings) {
    Parser::Map data;
    JST_CHECK(settings.serialize(data));

    std::string yaml;
    JST_CHECK(Parser::YamlEncode(data, yaml));

    return Platform::WriteFileAtomic(Platform::PathToUtf8(path), yaml);
}

Result Settings::Get(Settings& settings) {
    std::filesystem::path path;
    JST_CHECK(Impl::ResolvePath(path));

    Impl& impl = Impl::Instance();
    std::lock_guard lock(impl.mutex);
    if (!impl.loaded || impl.path != path) {
        Settings candidate;
        JST_CHECK(Impl::LoadFile(path, candidate));

        impl.settings = std::move(candidate);
        impl.path = std::move(path);
        impl.loaded = true;
    }

    settings = impl.settings;
    return Result::SUCCESS;
}

Result Settings::Set(const Settings& settings, bool persist) {
    std::filesystem::path path;
    JST_CHECK(Impl::ResolvePath(path));
    Settings candidate = settings;

    Impl& impl = Impl::Instance();
    std::lock_guard lock(impl.mutex);

    if (persist) {
        JST_CHECK(Impl::SaveFile(path, candidate));
    }

    impl.settings = std::move(candidate);
    impl.path = std::move(path);
    impl.loaded = true;

    return Result::SUCCESS;
}

}  // namespace Jetstream
