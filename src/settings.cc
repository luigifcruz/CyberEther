#include "jetstream/settings.hh"

#include <filesystem>
#include <fstream>

#include "jetstream/platform.hh"

namespace Jetstream {

namespace {

constexpr const char* SettingsFilename = "settings.yaml";

Result ValidateKey(const std::string& key) {
    if (key.empty()) {
        JST_ERROR("[SETTINGS] Setting key must not be empty.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result ResolveSettingsPath(std::filesystem::path& path) {
    std::string configPath;
    JST_CHECK(Platform::ConfigPath(configPath));

    path = std::filesystem::u8path(configPath) / SettingsFilename;
    return Result::SUCCESS;
}

std::filesystem::path ResolveTempSettingsPath(const std::filesystem::path& path) {
    auto tempPath = path;
    tempPath += ".tmp";
    return tempPath;
}

}  // namespace

Result Settings::LoadData(const std::string& key, Parser::Map& data) {
    JST_CHECK(ValidateKey(key));

    data.clear();

    std::filesystem::path path;
    JST_CHECK(ResolveSettingsPath(path));

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

    return Parser::YamlDecode(yaml, data);
}

Result Settings::SaveData(const std::string& key, const Parser::Map& data) {
    JST_CHECK(ValidateKey(key));

    std::filesystem::path path;
    JST_CHECK(ResolveSettingsPath(path));

    if (data.empty()) {
        std::error_code ec;
        (void)std::filesystem::remove(path, ec);
        if (ec) {
            JST_ERROR("[SETTINGS] Failed to remove settings file '{}'.", path.string());
            return Result::ERROR;
        }

        return Result::SUCCESS;
    }

    const auto parent = path.parent_path();
    if (!parent.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
        if (ec) {
            JST_ERROR("[SETTINGS] Cannot create directory '{}'.", parent.string());
            return Result::ERROR;
        }
    }

    std::string yaml;
    JST_CHECK(Parser::YamlEncode(data, yaml));

    const auto tempPath = ResolveTempSettingsPath(path);

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

Result Settings::Erase(const std::string& key) {
    Parser::Map data;
    JST_CHECK(LoadData(key, data));

    if (!data.contains(key)) {
        return Result::SUCCESS;
    }

    data.erase(key);
    return SaveData(key, data);
}

}  // namespace Jetstream
