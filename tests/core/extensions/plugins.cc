#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include <zlib.h>

#include "environment.hh"
#include "home_gallery_remote.hh"
#include "jetstream/config.hh"
#include "jetstream/logger.hh"
#include "jetstream/platform.hh"
#include "jetstream/plugin.hh"
#include "jetstream/registry.hh"
#include "plugin_gallery.hh"

#if defined(JETSTREAM_LOADER_OPENSSL_AVAILABLE)
#include <openssl/evp.h>
#endif

using namespace Jetstream;

namespace {

class TempDirectory {
 public:
    explicit TempDirectory(const std::string& label) {
        static std::atomic<std::uint64_t> sequence = 0;
        const auto parent = std::filesystem::temp_directory_path();

        for (std::uint64_t attempt = 0; attempt < 1024; ++attempt) {
            const auto timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
            const auto nonce = sequence.fetch_add(1, std::memory_order_relaxed);
            root = parent / Platform::PathFromUtf8(
                                "cyberether-plugin-test-" + label + "-" +
                                std::to_string(timestamp) + "-" + std::to_string(nonce));

            std::error_code ec;
            if (std::filesystem::create_directory(root, ec)) {
                return;
            }
            if (ec && ec != std::errc::file_exists) {
                throw std::runtime_error("failed to create plugin test directory");
            }
        }

        throw std::runtime_error("failed to allocate unique plugin test directory");
    }

    ~TempDirectory() {
        std::error_code ec;
        std::filesystem::remove_all(root, ec);
    }

    TempDirectory(const TempDirectory&) = delete;
    TempDirectory& operator=(const TempDirectory&) = delete;

    std::filesystem::path root;
};

using EnvironmentGuard = TestSupport::ScopedEnvironment;

class PluginSandbox {
 public:
    PluginSandbox()
        : directory_("sandbox"),
#if defined(_WIN32)
          environment_("LOCALAPPDATA"),
          secondary_("LOCALAPPDATA")
#elif defined(__APPLE__)
          environment_("CFFIXED_USER_HOME"),
          secondary_("CFFIXED_USER_HOME")
#else
          environment_("XDG_CACHE_HOME"),
          secondary_("XDG_DATA_HOME")
#endif
    {
#if defined(_WIN32) || defined(__APPLE__)
        const auto first = Platform::PathToUtf8(directory_.root);
        const auto second = first;
#else
        const auto first = Platform::PathToUtf8(directory_.root / "cache");
        const auto second = Platform::PathToUtf8(directory_.root / "data");
#endif
        if (!environment_.set(first) || !secondary_.set(second)) {
            throw std::runtime_error("failed to configure plugin sandbox");
        }

        for (const auto& resolve : {&Platform::CachePath, &Platform::DataPath}) {
            std::string path;
            if (resolve(path) != Result::SUCCESS) {
                throw std::runtime_error("failed to resolve plugin sandbox");
            }

            std::error_code ec;
            const auto relative = std::filesystem::relative(Platform::PathFromUtf8(path), directory_.root, ec);
            if (ec || relative.empty() || relative.is_absolute() || *relative.begin() == "..") {
                throw std::runtime_error("plugin storage escaped its test sandbox");
            }
        }
    }

 private:
    TempDirectory directory_;
    EnvironmentGuard environment_;
    EnvironmentGuard secondary_;
};

void EnsurePluginSandbox() {
#if !defined(JST_OS_BROWSER)
    static const PluginSandbox sandbox;
    (void)sandbox;
#endif
}

std::filesystem::path StoreRoot() {
    EnsurePluginSandbox();
    std::string path;
    if (Platform::DataPath(path) != Result::SUCCESS) {
        throw std::runtime_error("failed to resolve plugin store path");
    }
    return Platform::PathFromUtf8(path) / "plugins" / JETSTREAM_BUILD_ID;
}

std::size_t CountExtractionDirectories() {
    EnsurePluginSandbox();

    std::string cachePath;
    if (Platform::CachePath(cachePath) != Result::SUCCESS) {
        throw std::runtime_error("failed to resolve plugin cache path");
    }

    const auto runs = Platform::PathFromUtf8(cachePath) / "plugins" / "runs";
    std::error_code ec;
    if (!std::filesystem::exists(runs, ec)) {
        return 0;
    }

    std::size_t count = 0;
    for (const auto& run : std::filesystem::directory_iterator(runs, ec)) {
        for (const auto& entry : std::filesystem::directory_iterator(run.path(), ec)) {
            if (entry.is_directory(ec)) {
                ++count;
            }
        }
    }
    return count;
}

std::vector<std::uint8_t> ReadBytes(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("failed to read plugin test file");
    }
    return std::vector<std::uint8_t>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

void WriteBytes(const std::filesystem::path& path, const std::vector<std::uint8_t>& bytes) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) {
        throw std::runtime_error("failed to create plugin test file");
    }

    if (!bytes.empty()) {
        file.write(reinterpret_cast<const char*>(bytes.data()),
                   static_cast<std::streamsize>(bytes.size()));
    }
    if (!file) {
        throw std::runtime_error("failed to write plugin test file");
    }
}

std::vector<std::uint8_t> Deflate(const std::vector<std::uint8_t>& input) {
    if (input.size() > static_cast<std::size_t>(std::numeric_limits<uInt>::max())) {
        throw std::runtime_error("plugin test payload is too large");
    }

    z_stream stream = {};
    if (deflateInit2(&stream, Z_DEFAULT_COMPRESSION, Z_DEFLATED, -15, 8, Z_DEFAULT_STRATEGY) != Z_OK) {
        throw std::runtime_error("failed to initialize plugin test compressor");
    }

    stream.next_in = input.empty()
                         ? Z_NULL
                         : const_cast<Bytef*>(reinterpret_cast<const Bytef*>(input.data()));
    stream.avail_in = static_cast<uInt>(input.size());

    std::vector<std::uint8_t> output;
    std::array<std::uint8_t, 4096> chunk = {};
    int status = Z_OK;
    while (status != Z_STREAM_END) {
        stream.next_out = reinterpret_cast<Bytef*>(chunk.data());
        stream.avail_out = static_cast<uInt>(chunk.size());
        status = deflate(&stream, Z_FINISH);
        if (status != Z_OK && status != Z_STREAM_END) {
            deflateEnd(&stream);
            throw std::runtime_error("failed to compress plugin test payload");
        }

        const auto produced = chunk.size() - stream.avail_out;
        output.insert(output.end(), chunk.begin(), chunk.begin() + produced);
    }

    if (deflateEnd(&stream) != Z_OK) {
        throw std::runtime_error("failed to finish plugin test compression");
    }
    return output;
}

void PutU16(std::vector<std::uint8_t>& bytes, std::size_t offset, std::uint16_t value) {
    bytes.at(offset) = static_cast<std::uint8_t>(value);
    bytes.at(offset + 1) = static_cast<std::uint8_t>(value >> 8);
}

void PutU32(std::vector<std::uint8_t>& bytes, std::size_t offset, std::uint32_t value) {
    PutU16(bytes, offset, static_cast<std::uint16_t>(value));
    PutU16(bytes, offset + 2, static_cast<std::uint16_t>(value >> 16));
}

std::uint32_t GetU32(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
    return static_cast<std::uint32_t>(bytes.at(offset)) |
           (static_cast<std::uint32_t>(bytes.at(offset + 1)) << 8) |
           (static_cast<std::uint32_t>(bytes.at(offset + 2)) << 16) |
           (static_cast<std::uint32_t>(bytes.at(offset + 3)) << 24);
}

void AppendU16(std::vector<std::uint8_t>& bytes, std::uint16_t value) {
    bytes.resize(bytes.size() + 2);
    PutU16(bytes, bytes.size() - 2, value);
}

void AppendU32(std::vector<std::uint8_t>& bytes, std::uint32_t value) {
    bytes.resize(bytes.size() + 4);
    PutU32(bytes, bytes.size() - 4, value);
}

struct BundleEntry {
    std::string name;
    std::vector<std::uint8_t> content;
    std::uint32_t mode = 0100644;
    bool deflate = true;
};

BundleEntry TextEntry(const std::string& name, const std::string& content, std::uint32_t mode = 0100644) {
    return {name, std::vector<std::uint8_t>(content.begin(), content.end()), mode};
}

BundleEntry DirectoryEntry(const std::string& name) {
    return {name, {}, 040755, false};
}

std::vector<std::uint8_t> ZipArchive(const std::vector<BundleEntry>& entries) {
    std::vector<std::uint8_t> archive;
    std::vector<std::uint8_t> central;

    for (const auto& entry : entries) {
        const auto data = entry.deflate ? Deflate(entry.content) : entry.content;
        const auto method = static_cast<std::uint16_t>(entry.deflate ? 8 : 0);
        const auto crc = static_cast<std::uint32_t>(crc32(0L,
                                                          entry.content.empty() ? Z_NULL : entry.content.data(),
                                                          static_cast<uInt>(entry.content.size())));
        const auto offset = static_cast<std::uint32_t>(archive.size());
        const auto nameLength = static_cast<std::uint16_t>(entry.name.size());

        AppendU32(archive, 0x04034b50);
        AppendU16(archive, 20);
        AppendU16(archive, 0);
        AppendU16(archive, method);
        AppendU16(archive, 0);
        AppendU16(archive, 0x0021);
        AppendU32(archive, crc);
        AppendU32(archive, static_cast<std::uint32_t>(data.size()));
        AppendU32(archive, static_cast<std::uint32_t>(entry.content.size()));
        AppendU16(archive, nameLength);
        AppendU16(archive, 0);
        archive.insert(archive.end(), entry.name.begin(), entry.name.end());
        archive.insert(archive.end(), data.begin(), data.end());

        AppendU32(central, 0x02014b50);
        AppendU16(central, (3 << 8) | 20);
        AppendU16(central, 20);
        AppendU16(central, 0);
        AppendU16(central, method);
        AppendU16(central, 0);
        AppendU16(central, 0x0021);
        AppendU32(central, crc);
        AppendU32(central, static_cast<std::uint32_t>(data.size()));
        AppendU32(central, static_cast<std::uint32_t>(entry.content.size()));
        AppendU16(central, nameLength);
        AppendU16(central, 0);
        AppendU16(central, 0);
        AppendU16(central, 0);
        AppendU16(central, 0);
        AppendU32(central, entry.mode << 16);
        AppendU32(central, offset);
        central.insert(central.end(), entry.name.begin(), entry.name.end());
    }

    const auto centralOffset = static_cast<std::uint32_t>(archive.size());
    archive.insert(archive.end(), central.begin(), central.end());

    AppendU32(archive, 0x06054b50);
    AppendU16(archive, 0);
    AppendU16(archive, 0);
    AppendU16(archive, static_cast<std::uint16_t>(entries.size()));
    AppendU16(archive, static_cast<std::uint16_t>(entries.size()));
    AppendU32(archive, static_cast<std::uint32_t>(central.size()));
    AppendU32(archive, centralOffset);
    AppendU16(archive, 0);
    return archive;
}

std::size_t CentralHeaderOffset(const std::vector<std::uint8_t>& archive) {
    return GetU32(archive, archive.size() - 22 + 16);
}

std::string LibraryExtension() {
#if defined(JST_OS_MAC) || defined(JST_OS_IOS)
    return ".dylib";
#elif defined(JST_OS_WINDOWS)
    return ".dll";
#else
    return ".so";
#endif
}

std::string LibraryPath(const std::string& id) {
    return "lib/" + id + LibraryExtension();
}

struct ManifestSpec {
    std::string id = "fixture";
    std::string title = "Fixture";
    std::string version = "1.0.0";
    std::string cyberether = JETSTREAM_VERSION_STR;
    std::string build = JETSTREAM_BUILD_ID;
    std::string extra;
};

std::string Manifest(const ManifestSpec& spec) {
    std::string manifest;
    if (!spec.id.empty()) {
        manifest += "id: \"" + spec.id + "\"\n";
    }
    if (!spec.title.empty()) {
        manifest += "title: \"" + spec.title + "\"\n";
    }
    if (!spec.version.empty()) {
        manifest += "version: \"" + spec.version + "\"\n";
    }
    if (!spec.cyberether.empty()) {
        manifest += "cyberether: \"" + spec.cyberether + "\"\n";
    }
    if (!spec.build.empty()) {
        manifest += "build: \"" + spec.build + "\"\n";
    }
    return manifest + spec.extra;
}

void WriteBundle(const std::filesystem::path& path, const ManifestSpec& spec, std::vector<BundleEntry> entries = {}) {
    entries.insert(entries.begin(), TextEntry("manifest.yml", Manifest(spec)));
    WriteBytes(path, ZipArchive(entries));
}

void WriteGarbageBundle(const std::filesystem::path& path,
                        const ManifestSpec& spec,
                        std::vector<BundleEntry> extra = {}) {
    extra.insert(extra.begin(), TextEntry(LibraryPath(spec.id), "not a library"));
    WriteBundle(path, spec, std::move(extra));
}

std::optional<Plugin::Info> Find(const std::string& id) {
    for (auto& plugin : Plugin::List()) {
        if (plugin.id == id) {
            return plugin;
        }
    }
    return std::nullopt;
}

void RequireLoadRejected(const std::string& path) {
    EnsurePluginSandbox();
    const auto count = Plugin::List().size();
    JST_LOG_LAST_ERROR().clear();

    REQUIRE(Plugin::Load(path) == Result::ERROR);
    REQUIRE(Plugin::List().size() == count);
}

void RequireInstallRejected(const std::string& path) {
    EnsurePluginSandbox();
    const auto count = Plugin::List().size();
    JST_LOG_LAST_ERROR().clear();

    std::string id = "stale";
    REQUIRE(Plugin::Install(path, id) == Result::ERROR);
    REQUIRE(id.empty());
    REQUIRE(Plugin::List().size() == count);
}

void RequireRejected(const std::filesystem::path& path, const std::string& message = {}) {
    const auto utf8 = Platform::PathToUtf8(path);
    RequireLoadRejected(utf8);
    if (!message.empty()) {
        CHECK(JST_LOG_LAST_ERROR().find(message) != std::string::npos);
    }
#if !defined(JST_OS_BROWSER)
    RequireInstallRejected(utf8);
    if (!message.empty()) {
        CHECK(JST_LOG_LAST_ERROR().find(message) != std::string::npos);
    }
#endif
}

}  // namespace

TEST_CASE("Plugin entry points reject invalid bundle paths", "[core][extensions][plugin]") {
    EnsurePluginSandbox();
    SECTION("empty paths") {
        RequireLoadRejected("");
        RequireInstallRejected("");
    }

    SECTION("non-bundle extensions") {
        RequireLoadRejected("cyberether-extension.so");
        RequireInstallRejected("cyberether-extension.tar.gz");
    }

    SECTION("missing bundles") {
        const TempDirectory temp("missing");
        RequireRejected(temp.root / "missing.cep");
    }

    SECTION("bundle paths that are directories") {
        const TempDirectory temp("directory");
        const auto directory = temp.root / "directory.cep";
        std::error_code ec;
        REQUIRE(std::filesystem::create_directory(directory, ec));
        REQUIRE_FALSE(ec);
        RequireRejected(directory);
    }

    SECTION("unknown plugin IDs") {
        EnsurePluginSandbox();
        CHECK(Plugin::Path("nothere").empty());
        CHECK(Plugin::Owner("nothere_block", "generic").empty());
#if !defined(JST_OS_BROWSER)
        CHECK(Plugin::Remove("nothere") == Result::ERROR);
#endif
        CHECK(Plugin::Reload("nothere") == Result::ERROR);
    }
}

TEST_CASE("Plugin loader rejects malformed CEP zip structure", "[core][extensions][plugin]") {
    EnsurePluginSandbox();
    const TempDirectory temp("archive");
    const auto archivePath = temp.root / "fixture.cep";
    const auto manifest = TextEntry("manifest.yml", "id: fixture\n");

    SECTION("empty file") {
        WriteBytes(archivePath, {});
        RequireRejected(archivePath, "Failed to read plugin bundle");
    }

    SECTION("non-zip bytes") {
        WriteBytes(archivePath, {'n', 'o', 't', '-', 'a', '-', 'z', 'i', 'p'});
        RequireRejected(archivePath, "Failed to read plugin bundle");
    }

    SECTION("gzip bytes from the old bundle format") {
        WriteBytes(archivePath, {0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03});
        RequireRejected(archivePath, "Failed to read plugin bundle");
    }

    SECTION("truncated archive") {
        auto archive = ZipArchive({manifest});
        archive.resize(archive.size() - 4);
        WriteBytes(archivePath, archive);
        RequireRejected(archivePath, "Failed to read plugin bundle");
    }

    SECTION("archive truncated inside the central directory") {
        auto archive = ZipArchive({manifest});
        const auto end = std::vector<std::uint8_t>(archive.end() - 22, archive.end());
        archive.resize(CentralHeaderOffset(archive) + 10);
        archive.insert(archive.end(), end.begin(), end.end());
        WriteBytes(archivePath, archive);
        RequireRejected(archivePath, "Failed to read plugin bundle");
    }

    SECTION("too many entries") {
        std::vector<BundleEntry> entries = {manifest};
        for (int i = 0; i < 10000; ++i) {
            entries.push_back({"data/" + std::to_string(i), {}, 0100644, false});
        }
        WriteBytes(archivePath, ZipArchive(entries));
        RequireRejected(archivePath, "more than 10000 entries");
    }

    SECTION("encrypted entry") {
        auto archive = ZipArchive({manifest});
        PutU16(archive, 6, 0x0001);
        PutU16(archive, CentralHeaderOffset(archive) + 8, 0x0001);
        WriteBytes(archivePath, archive);
        RequireRejected(archivePath, "Failed to extract bundle entry 'manifest.yml'");
    }

    SECTION("unsupported compression method") {
        auto archive = ZipArchive({manifest});
        PutU16(archive, 8, 12);
        PutU16(archive, CentralHeaderOffset(archive) + 10, 12);
        WriteBytes(archivePath, archive);
        RequireRejected(archivePath, "Failed to extract bundle entry 'manifest.yml'");
    }

    SECTION("corrupted local header") {
        auto archive = ZipArchive({manifest});
        archive.at(0) ^= 0x01;
        WriteBytes(archivePath, archive);
        RequireRejected(archivePath, "Failed to extract bundle entry 'manifest.yml'");
    }

    SECTION("CRC mismatch") {
        auto archive = ZipArchive({manifest});
        const auto offset = CentralHeaderOffset(archive) + 16;
        PutU32(archive, offset, GetU32(archive, offset) ^ 0x1);
        WriteBytes(archivePath, archive);
        RequireRejected(archivePath, "decompression failed or archive is corrupted");
    }

    SECTION("stored entry CRC mismatch") {
        auto entry = manifest;
        entry.deflate = false;
        auto archive = ZipArchive({entry});
        archive.at(30 + entry.name.size()) ^= 0x01;
        WriteBytes(archivePath, archive);
        RequireRejected(archivePath, "decompression failed or archive is corrupted");
    }

    SECTION("entry larger than its declared size") {
        auto archive = ZipArchive({manifest});
        PutU32(archive, CentralHeaderOffset(archive) + 24, 4);
        WriteBytes(archivePath, archive);
        RequireRejected(archivePath, "Failed to extract bundle entry 'manifest.yml'");
    }

    SECTION("entry smaller than its declared size") {
        auto archive = ZipArchive({manifest});
        PutU32(archive, CentralHeaderOffset(archive) + 24, 64);
        WriteBytes(archivePath, archive);
        RequireRejected(archivePath, "Failed to extract bundle entry 'manifest.yml'");
    }

    SECTION("file entry carrying the DOS directory attribute") {
        const auto library = TextEntry(LibraryPath("fixture"), "library");
        auto archive = ZipArchive({TextEntry("manifest.yml", Manifest(ManifestSpec{})),
                                   library,
                                   TextEntry("data/table.bin", "table")});
        const auto third = CentralHeaderOffset(archive) + 2 * 46 + manifest.name.size() + library.name.size();
        PutU16(archive, third + 4, 20);
        PutU32(archive, third + 38, 0x10);
        WriteBytes(archivePath, archive);
        RequireRejected(archivePath, "Bundle entry 'data/table.bin' is smaller than its declared size");
    }

    SECTION("bundle expanding past 2 GiB") {
        auto archive = ZipArchive({manifest, TextEntry("data/big", "x")});
        const auto second = CentralHeaderOffset(archive) + 46 + manifest.name.size();
        PutU32(archive, second + 24, 0x80000000);
        WriteBytes(archivePath, archive);
        RequireRejected(archivePath, "more than 2 GiB");
    }

    SECTION("manifest larger than 64 KiB") {
        WriteBytes(archivePath, ZipArchive({TextEntry("manifest.yml", "id: fixture\n" + std::string(65536, '#'))}));
        RequireRejected(archivePath, "larger than 64 KiB");
    }

    SECTION("unsafe archive paths") {
        for (const auto& name : {"../manifest.yml", "/manifest.yml", "lib//fixture", "lib\\fixture", "C:/lib/fixture",
                                 "lib/../manifest.yml", "/", "lib//", "lib/./fixture", "lib/fixture.", "lib/fixture ",
                                 "lib/con", "data/NUL.txt", "examples/Com1.yml", "data/COM\xc2\xb9.txt",
                                 "data/NUL .txt"}) {
            CAPTURE(name);
            WriteBytes(archivePath, ZipArchive({manifest, TextEntry(name, "x")}));
            RequireRejected(archivePath, "Unsafe bundle entry path");
        }
    }

    SECTION("duplicate entries") {
        WriteBytes(archivePath, ZipArchive({manifest, TextEntry("data/a", "1"), TextEntry("data/a", "2")}));
        RequireRejected(archivePath, "Duplicate bundle entry 'data/a'");
    }

    SECTION("entries differing only in case") {
        WriteBytes(archivePath,
                   ZipArchive({manifest, TextEntry("data/Table.bin", "1"), TextEntry("data/table.bin", "2")}));
        RequireRejected(archivePath, "Duplicate bundle entry 'data/table.bin'");
    }

    SECTION("directory entry colliding with a file") {
        WriteBytes(archivePath, ZipArchive({manifest, TextEntry("data/a", "1"), DirectoryEntry("data/a/")}));
        RequireRejected(archivePath, "Duplicate bundle entry 'data/a/'");
    }

    SECTION("empty zip") {
        WriteBytes(archivePath, ZipArchive({}));
        RequireRejected(archivePath, "does not contain a manifest");
    }

    SECTION("failed extraction leaves nothing behind") {
        const auto extractionCount = CountExtractionDirectories();
        auto archive = ZipArchive({manifest});
        const auto offset = CentralHeaderOffset(archive) + 16;
        PutU32(archive, offset, GetU32(archive, offset) ^ 0x1);
        WriteBytes(archivePath, archive);
        RequireRejected(archivePath, "decompression failed or archive is corrupted");
        REQUIRE(CountExtractionDirectories() == extractionCount);
    }
}

TEST_CASE("Plugin manifests are validated before any library opens", "[core][extensions][plugin]") {
    EnsurePluginSandbox();
    const TempDirectory temp("manifest");
    const auto archivePath = temp.root / "fixture.cep";

    SECTION("missing manifest") {
        WriteBytes(archivePath, ZipArchive({TextEntry("data/manifest.yml", "id: fixture\n")}));
        RequireRejected(archivePath, "does not contain a manifest");
    }

    SECTION("entries outside the bundle layout") {
        for (const auto& name : {"other.yml", "builds/fixture.so", "manifest.yml/nested", ".users/lock"}) {
            CAPTURE(name);
            WriteBundle(archivePath, ManifestSpec{}, {TextEntry(name, "x")});
            RequireRejected(archivePath, "Unexpected bundle entry");
        }
    }

    SECTION("malformed YAML") {
        WriteBytes(archivePath, ZipArchive({TextEntry("manifest.yml", "id: [\n")}));
        RequireRejected(archivePath);
    }

    SECTION("old manifests with a metadata section") {
        WriteBytes(archivePath, ZipArchive({TextEntry("manifest.yml",
                                                      "metadata:\n  name: fixture\n  version: 1.0.0\n")}));
        RequireRejected(archivePath, "unknown key 'metadata'");
    }

    SECTION("unknown keys") {
        auto spec = ManifestSpec{};
        spec.extra = "homepage: https://example.org\n";
        WriteGarbageBundle(archivePath, spec);
        RequireRejected(archivePath, "unknown key 'homepage'");
    }

    SECTION("invalid plugin IDs") {
        for (const auto& id : {"", "Fixture", "1fixture", "fix_ture", "fix-ture",
                               "abcdefghijklmnopqrstuvwxyz0123456"}) {
            CAPTURE(id);
            auto spec = ManifestSpec{};
            spec.id = id;
            WriteGarbageBundle(archivePath, spec);
            RequireRejected(archivePath, "is invalid");
        }
        for (const auto& id : {"aux", "com1", "lpt9"}) {
            CAPTURE(id);
            auto spec = ManifestSpec{};
            spec.id = id;
            WriteGarbageBundle(archivePath, spec);
            RequireRejected(archivePath, "Unsafe bundle entry path 'lib/" + std::string(id));
        }
    }

    SECTION("missing title") {
        auto spec = ManifestSpec{};
        spec.title.clear();
        WriteGarbageBundle(archivePath, spec);
        RequireRejected(archivePath, "does not have a title");
    }

    SECTION("invalid plugin versions") {
        for (const auto& version : {"", ".1.0", "1..0", "../1", "1.0/2", "1.0 2"}) {
            CAPTURE(version);
            auto spec = ManifestSpec{};
            spec.version = version;
            WriteGarbageBundle(archivePath, spec);
            RequireRejected(archivePath, "invalid version");
        }
    }

    SECTION("other CyberEther versions") {
        for (const auto& version : {"", "0.0.1", "1.20"}) {
            CAPTURE(version);
            auto spec = ManifestSpec{};
            spec.cyberether = version;
            WriteGarbageBundle(archivePath, spec);
            RequireRejected(archivePath, "was built for CyberEther");
        }
    }

    SECTION("other CyberEther builds") {
        for (const auto& build : {"", "other-build", JETSTREAM_BUILD_ID "-cuda"}) {
            CAPTURE(build);
            auto spec = ManifestSpec{};
            spec.build = build;
            WriteGarbageBundle(archivePath, spec);
            RequireRejected(archivePath, "but this CyberEther is " JETSTREAM_BUILD_ID);
        }
    }

    SECTION("libraries with another name") {
        WriteBundle(archivePath, ManifestSpec{}, {TextEntry("lib/other" + LibraryExtension(), "library")});
        RequireRejected(archivePath, "does not include lib/fixture");
    }
}

#if !defined(JST_OS_BROWSER)

TEST_CASE("Plugin installs keep their own copy", "[core][extensions][plugin]") {
    EnsurePluginSandbox();
    const TempDirectory temp("install");
    const auto archivePath = temp.root / "stored.cep";

    auto spec = ManifestSpec{};
    spec.id = "stored";
    spec.title = "Stored";
    spec.extra = "summary: \"Kept in the store.\"\nauthor: \"Tester\"\nlicense: \"MIT\"\n";
    auto table = TextEntry("data/table.bin", "table");
    table.deflate = false;
    WriteGarbageBundle(archivePath, spec, {
        DirectoryEntry("data/"),
        table,
        DirectoryEntry("examples/"),
        TextEntry("lib/helper", "#!/bin/sh\n", 0100744),
    });

    std::string id;
    REQUIRE(Plugin::Install(Platform::PathToUtf8(archivePath), id) == Result::SUCCESS);
    CHECK(id == "stored");

    std::filesystem::remove(archivePath);

    const auto plugin = Find("stored");
    REQUIRE(plugin.has_value());
    CHECK(plugin->title == "Stored");
    CHECK(plugin->summary == "Kept in the store.");
    CHECK(plugin->author == "Tester");
    CHECK(plugin->license == "MIT");
    CHECK(plugin->version == "1.0.0");
    CHECK(plugin->cyberether == JETSTREAM_VERSION_STR);
    CHECK_FALSE(plugin->temporary);
    CHECK(plugin->status == Plugin::Status::Restart);
    CHECK(Plugin::Path("stored").empty());

    const auto folder = Platform::PathFromUtf8(plugin->path);
    const auto relative = std::filesystem::relative(folder, StoreRoot());
    CHECK(*relative.begin() == "stored");
    CHECK(std::filesystem::is_regular_file(folder / "manifest.yml"));
    CHECK(std::filesystem::is_regular_file(folder / "data" / "table.bin"));
    CHECK(std::filesystem::is_regular_file(folder / Platform::PathFromUtf8(LibraryPath("stored"))));
#if !defined(JST_OS_WINDOWS)
    const auto permissions = std::filesystem::status(folder / "lib" / "helper").permissions();
    CHECK((permissions & std::filesystem::perms::owner_exec) != std::filesystem::perms::none);
    CHECK((permissions & std::filesystem::perms::others_exec) == std::filesystem::perms::none);
    const auto data = std::filesystem::status(folder / "data" / "table.bin").permissions();
    CHECK((data & std::filesystem::perms::owner_exec) == std::filesystem::perms::none);
#endif

    REQUIRE(Plugin::Remove("stored") == Result::SUCCESS);
    CHECK_FALSE(Find("stored").has_value());
    CHECK_FALSE(std::filesystem::exists(folder));
    CHECK(Plugin::Remove("stored") == Result::ERROR);
    CHECK(Plugin::LoadInstalled() == Result::SUCCESS);

    REQUIRE(Plugin::Install(Platform::PathToUtf8(temp.root / "again.cep"), id) == Result::ERROR);
    WriteGarbageBundle(temp.root / "again.cep", spec);
    REQUIRE(Plugin::Install(Platform::PathToUtf8(temp.root / "again.cep"), id) == Result::SUCCESS);
    const auto kept = Platform::PathFromUtf8(Find("stored")->path);
    {
        std::ofstream index(StoreRoot() / "index.yml", std::ios::binary | std::ios::trunc);
    }
    CHECK(Plugin::LoadInstalled() == Result::ERROR);
    CHECK(std::filesystem::exists(kept));
    for (const auto* corrupted : {"{}\n", "plugins:\n  - id: stored\n",
                                  "plugins:\n  - id: stored\n    folder: a\n  - id: stored\n    folder: b\n"}) {
        std::ofstream index(StoreRoot() / "index.yml", std::ios::binary | std::ios::trunc);
        index << corrupted;
        index.close();
        CHECK(Plugin::LoadInstalled() == Result::ERROR);
        CHECK(std::filesystem::exists(kept));
    }
    std::filesystem::remove(StoreRoot() / "index.yml");
    CHECK(Plugin::LoadInstalled() == Result::ERROR);
    CHECK(Plugin::Install(Platform::PathToUtf8(temp.root / "again.cep"), id) == Result::ERROR);
    CHECK(std::filesystem::exists(kept));
    {
        std::ofstream index(StoreRoot() / "index.yml", std::ios::binary | std::ios::trunc);
        index << "plugins: []\n";
    }
    CHECK(Plugin::LoadInstalled() == Result::SUCCESS);
}

TEST_CASE("Installed plugins that fail to load report a failed status", "[core][extensions][plugin]") {
    EnsurePluginSandbox();
    const TempDirectory temp("failed");
    const auto archivePath = temp.root / "broken.cep";

    auto spec = ManifestSpec{};
    spec.id = "broken";
    WriteGarbageBundle(archivePath, spec);

    std::string id;
    REQUIRE(Plugin::Install(Platform::PathToUtf8(archivePath), id) == Result::SUCCESS);
    REQUIRE(Plugin::LoadInstalled() == Result::SUCCESS);

    const auto plugin = Find("broken");
    REQUIRE(plugin.has_value());
    CHECK(plugin->status == Plugin::Status::Failed);
    CHECK(Plugin::Path("broken").empty());

    REQUIRE(Plugin::Remove("broken") == Result::SUCCESS);
    CHECK_FALSE(Find("broken").has_value());
}

#endif

#if defined(JETSTREAM_TEST_PLUGIN_MODULE)

TEST_CASE("Plugins loaded for one run register prefixed blocks and know their path", "[core][extensions][plugin]") {
    EnsurePluginSandbox();
    const TempDirectory temp("fixture");
    const auto archivePath = temp.root / "fixture.cep";
    WriteBundle(archivePath, ManifestSpec{}, {
        {LibraryPath("fixture"), ReadBytes(JETSTREAM_TEST_PLUGIN_MODULE)},
        TextEntry("examples/demo.yml", "version: 2\ntitle: Demo\ngraph: []\n"),
        TextEntry("examples/notes.txt", "ignored"),
    });

    REQUIRE(Plugin::Load(Platform::PathToUtf8(archivePath)) == Result::SUCCESS);
    REQUIRE(Plugin::Load(Platform::PathToUtf8(archivePath)) == Result::SUCCESS);

    const auto plugin = Find("fixture");
    REQUIRE(plugin.has_value());
    CHECK(plugin->temporary);
    CHECK(plugin->status == Plugin::Status::Loaded);
    CHECK(plugin->blocks == std::vector<std::string>{"fixture_probe"});
    CHECK(plugin->examples == std::vector<std::string>{"fixture/demo"});

    const auto root = Plugin::Path("fixture");
    REQUIRE_FALSE(root.empty());
    CHECK(Platform::PathToUtf8(root) == plugin->path);

    {
        const auto blocks = Registry::ListAvailableBlocks("fixture_probe");
        REQUIRE(blocks.size() == 1);
        CHECK(blocks.front().summary == Platform::PathToUtf8(root));
    }

    CHECK(Plugin::Owner("fixture_probe", "generic") == "fixture");
    CHECK(Plugin::Owner("fft", "generic").empty());

    REQUIRE(Plugin::Reload("fixture") == Result::SUCCESS);
    const auto reloaded = Plugin::Path("fixture");
    CHECK_FALSE(reloaded.empty());
    CHECK(reloaded != root);
    CHECK_FALSE(std::filesystem::exists(root));
    CHECK(Registry::ListAvailableBlocks("fixture_probe").size() == 1);
}

TEST_CASE("Plugins that register blocks without their prefix are rejected", "[core][extensions][plugin]") {
    EnsurePluginSandbox();
    const TempDirectory temp("loose");
    const auto archivePath = temp.root / "loose.cep";
    auto spec = ManifestSpec{};
    spec.id = "loose";
    WriteBundle(archivePath, spec, {{LibraryPath("loose"), ReadBytes(JETSTREAM_TEST_PLUGIN_UNPREFIXED_MODULE)}});

    RequireLoadRejected(Platform::PathToUtf8(archivePath));
    CHECK(JST_LOG_LAST_ERROR().find("without the 'loose_' prefix") != std::string::npos);
    CHECK(Registry::ListAvailableBlocks("probe").empty());
    CHECK(Plugin::Path("loose").empty());
}

TEST_CASE("Installed plugins load at startup and defer updates to the next start", "[core][extensions][plugin]") {
    EnsurePluginSandbox();
    const TempDirectory temp("installed");
    const auto module = ReadBytes(JETSTREAM_TEST_PLUGIN_INSTALLED_MODULE);
    auto spec = ManifestSpec{};
    spec.id = "installed";

    const auto first = temp.root / "installed-1.cep";
    WriteBundle(first, spec, {{LibraryPath("installed"), module}});

    std::string id;
    REQUIRE(Plugin::Install(Platform::PathToUtf8(first), id) == Result::SUCCESS);
    REQUIRE(Plugin::LoadInstalled() == Result::SUCCESS);

    auto plugin = Find("installed");
    REQUIRE(plugin.has_value());
    CHECK(plugin->status == Plugin::Status::Loaded);
    CHECK_FALSE(plugin->temporary);
    CHECK(plugin->pending.empty());
    CHECK(Registry::ListAvailableBlocks("installed_probe").size() == 1);
    CHECK(Plugin::Reload("installed") == Result::ERROR);

    spec.version = "2.0.0";
    const auto second = temp.root / "installed-2.cep";
    WriteBundle(second, spec, {{LibraryPath("installed"), module}});
    REQUIRE(Plugin::Install(Platform::PathToUtf8(second), id) == Result::SUCCESS);
    REQUIRE(Plugin::LoadInstalled() == Result::SUCCESS);

    plugin = Find("installed");
    REQUIRE(plugin.has_value());
    CHECK(plugin->status == Plugin::Status::Loaded);
    CHECK(plugin->version == "1.0.0");
    CHECK(plugin->pending == "2.0.0");

    REQUIRE(Plugin::Remove("installed") == Result::SUCCESS);
    CHECK_FALSE(Find("installed").has_value());
    CHECK(Registry::ListAvailableBlocks("installed_probe").size() == 1);

    REQUIRE(Plugin::Install(Platform::PathToUtf8(second), id) == Result::SUCCESS);
    plugin = Find("installed");
    REQUIRE(plugin.has_value());
    CHECK(plugin->status == Plugin::Status::Loaded);
    CHECK(plugin->version == "1.0.0");
    CHECK(plugin->pending == "2.0.0");

    REQUIRE(Plugin::Remove("installed") == Result::SUCCESS);
}

#endif

#if !defined(JST_OS_BROWSER)

namespace {

class FakeGallery {
 public:
    std::map<std::string, std::string> files;
    std::vector<std::string> requests;

    void list(const std::string& body, const Plugin::Gallery::Query& query) {
        files[std::string(Home::Gallery::Remote::Host) + Plugin::Gallery::Path(query)] = body;
    }

    Plugin::Gallery::Source source() {
        return {
            .send = [this](const Platform::Http::Client::Request& request) {
                requests.push_back(request.host + request.path);
                Platform::Http::Client::Response response;
                const auto file = files.find(request.host + request.path);
                response.status = file == files.end() ? 404 : 200;
                if (request.onResponse && !request.onResponse(response.status, {})) {
                    return response;
                }
                if (file == files.end()) {
                    return response;
                }
                if (request.onData) {
                    request.onData(file->second);
                } else {
                    response.body = file->second;
                }
                return response;
            },
        };
    }
};

std::string Entry(const std::string& id,
                  const std::string& version,
                  const std::string& url,
                  const std::string& digest,
                  std::uint64_t size) {
    return "{\"id\":\"" + id + "\",\"title\":\"Gallery " + id + "\",\"summary\":\"From the gallery.\",\"version\":\"" +
           version + "\",\"url\":\"" + url + "\",\"size\":" + std::to_string(size) + ",\"cepSha256\":\"" +
           digest + "\"}";
}

std::string Listing(const std::vector<std::string>& entries) {
    std::string body = "{\"plugins\":[";
    for (std::size_t i = 0; i < entries.size(); ++i) {
        body += (i == 0 ? "" : ",") + entries[i];
    }
    return body + "]}";
}

#if defined(JETSTREAM_LOADER_OPENSSL_AVAILABLE)
std::string Sha256(const std::string& data) {
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int length = 0;
    EVP_Digest(data.data(), data.size(), digest, &length, EVP_sha256(), nullptr);

    constexpr const char* kHexDigits = "0123456789abcdef";
    std::string hex;
    for (unsigned int i = 0; i < length; ++i) {
        hex += kHexDigits[digest[i] >> 4];
        hex += kHexDigits[digest[i] & 0x0F];
    }
    return hex;
}
#endif

std::string BundleBytes(const std::filesystem::path& directory, const ManifestSpec& spec) {
    const auto path = directory / (spec.id + "-" + spec.version + ".cep");
    WriteGarbageBundle(path, spec);
    const auto bytes = ReadBytes(path);
    return std::string(bytes.begin(), bytes.end());
}

}  // namespace

TEST_CASE("Plugin gallery queries mirror the flowgraph gallery", "[core][extensions][plugin][gallery]") {
    const auto base = Home::Gallery::Remote::Current();
    const auto query = Plugin::Gallery::Current({"alpha", "beta"});
    CHECK(query.owner == base.owner);
    CHECK(query.version == base.version);
    CHECK(query.platform == base.platform);
    CHECK(query.arch == base.arch);
    CHECK(query.build == JETSTREAM_BUILD_ID);

    CHECK(Plugin::Gallery::Path(query) ==
          Home::Gallery::Remote::Path(base).replace(0, std::string("/api/v1/flowgraphs").size(), "/api/v1/plugins") +
              "&build=" JETSTREAM_BUILD_ID "&ids=alpha,beta");
    CHECK(Plugin::Gallery::Path({}) == "/api/v1/plugins");
    CHECK(Plugin::Gallery::Path({.version = "1.2.3"}) == "/api/v1/plugins?version=1.2.3");

    {
        EnvironmentGuard host(Plugin::Gallery::HostVariable);
        REQUIRE(host.set("http://127.0.0.1:8899"));
        CHECK(Plugin::Gallery::Host() == "http://127.0.0.1:8899");
        REQUIRE(host.set(""));
        CHECK(Plugin::Gallery::Host() == Home::Gallery::Remote::Host);
    }

    CHECK(Plugin::Gallery::ValidId("blueprint"));
    CHECK_FALSE(Plugin::Gallery::ValidId(""));
    CHECK_FALSE(Plugin::Gallery::ValidId("Blue_print"));
    CHECK_FALSE(Plugin::Gallery::ValidId("con"));
    CHECK_FALSE(Plugin::Gallery::ValidId("https://cyberether.org/gallery/plugins/blueprint"));
}

TEST_CASE("Plugin gallery lists use the flat entry format", "[core][extensions][plugin][gallery]") {
    FakeGallery gallery;
    const auto query = Plugin::Gallery::Current();
    gallery.list(Listing({
        Entry("alpha", "1.0.0", "https://files.example/alpha.cep", std::string(64, 'a'), 10),
        "{\"id\":\"waiting\"}",
        "{\"id\":\"Bad_Id\",\"version\":\"1.0.0\"}",
        "\"not an object\"",
        Entry("alpha", "2.0.0", "https://files.example/alpha2.cep", std::string(64, 'b'), 20),
    }), query);

    Plugin::Gallery::Entries entries;
    REQUIRE(Plugin::Gallery::Fetch(gallery.source(), query, {}, entries) == Result::SUCCESS);
    REQUIRE(entries.size() == 2);
    CHECK(entries[0].id == "alpha");
    CHECK(entries[0].title == "Gallery alpha");
    CHECK(entries[0].version == "1.0.0");
    CHECK(entries[0].size == 10);
    CHECK(entries[0].available());
    CHECK(entries[1].id == "waiting");
    CHECK(entries[1].title == "waiting");
    CHECK_FALSE(entries[1].available());

    gallery.list("{\"plugins\": 3}", query);
    CHECK(Plugin::Gallery::Fetch(gallery.source(), query, {}, entries) == Result::ERROR);
    gallery.list("not json", query);
    CHECK(Plugin::Gallery::Fetch(gallery.source(), query, {}, entries) == Result::ERROR);
    CHECK(Plugin::Gallery::Fetch(gallery.source(), Plugin::Gallery::Current({"other"}), {}, entries) == Result::ERROR);
    CHECK(Plugin::Gallery::Fetch({}, query, {}, entries) == Result::ERROR);
}

#if defined(JETSTREAM_LOADER_OPENSSL_AVAILABLE)

TEST_CASE("Plugin gallery downloads fail closed on size and checksum", "[core][extensions][plugin][gallery]") {
    const TempDirectory temp("download");
    const auto destination = temp.root / "bundle.cep";
    const std::string bytes = "bundle bytes";

    FakeGallery gallery;
    gallery.files["https://files.example/bundle.cep"] = bytes;

    Plugin::Gallery::Entry entry{
        .id = "alpha",
        .version = "1.0.0",
        .url = "https://files.example/bundle.cep",
        .size = bytes.size(),
        .digest = Sha256(bytes),
    };
    REQUIRE(Plugin::Gallery::Download(gallery.source(), entry, destination, {}) == Result::SUCCESS);
    const auto written = ReadBytes(destination);
    CHECK(std::string(written.begin(), written.end()) == bytes);

    SECTION("wrong checksum") {
        entry.digest = std::string(64, '0');
        CHECK(Plugin::Gallery::Download(gallery.source(), entry, destination, {}) == Result::ERROR);
        CHECK(JST_LOG_LAST_ERROR().find("does not match its checksum") != std::string::npos);
        CHECK_FALSE(std::filesystem::exists(destination));
    }

    SECTION("wrong size") {
        entry.size = bytes.size() + 1;
        CHECK(Plugin::Gallery::Download(gallery.source(), entry, destination, {}) == Result::ERROR);
        CHECK_FALSE(std::filesystem::exists(destination));
    }

    SECTION("larger than listed") {
        entry.size = bytes.size() - 1;
        CHECK(Plugin::Gallery::Download(gallery.source(), entry, destination, {}) == Result::ERROR);
        CHECK_FALSE(std::filesystem::exists(destination));
    }

    SECTION("missing checksum") {
        entry.digest.clear();
        CHECK(Plugin::Gallery::Download(gallery.source(), entry, destination, {}) == Result::ERROR);
        CHECK(JST_LOG_LAST_ERROR().find("no valid checksum") != std::string::npos);
    }

    SECTION("missing file") {
        entry.url = "https://files.example/missing.cep";
        CHECK(Plugin::Gallery::Download(gallery.source(), entry, destination, {}) == Result::ERROR);
        CHECK_FALSE(std::filesystem::exists(destination));
    }
}

TEST_CASE("Gallery installs remember their source and follow gallery updates", "[core][extensions][plugin][gallery]") {
    EnsurePluginSandbox();
    const TempDirectory temp("gallery");
    FakeGallery gallery;
    const auto query = Plugin::Gallery::Current({"remote"});

    auto spec = ManifestSpec{};
    spec.id = "remote";
    const auto first = BundleBytes(temp.root, spec);
    gallery.files["https://files.example/remote-1.cep"] = first;
    gallery.list(Listing({Entry("remote", "1.0.0", "https://files.example/remote-1.cep", Sha256(first), first.size())}),
                 query);

    REQUIRE(Plugin::Gallery::Install(gallery.source(), "remote") == Result::SUCCESS);

    auto plugin = Find("remote");
    REQUIRE(plugin.has_value());
    CHECK(plugin->gallery);
    CHECK(plugin->version == "1.0.0");
    CHECK(plugin->status == Plugin::Status::Restart);

    Plugin::Gallery::Entries updates;
    REQUIRE(Plugin::Gallery::Updates(gallery.source(), updates) == Result::SUCCESS);
    CHECK(updates.empty());

    std::vector<std::string> updated;
    REQUIRE(Plugin::Gallery::Update(gallery.source(), {"remote"}, updated) == Result::SUCCESS);
    CHECK(updated.empty());

    spec.version = "2.0.0";
    const auto second = BundleBytes(temp.root, spec);
    gallery.files["https://files.example/remote-2.cep"] = second;
    gallery.list(Listing({Entry("remote", "2.0.0", "https://files.example/remote-2.cep", Sha256(second), second.size())}),
                 query);
    REQUIRE(Plugin::Gallery::Updates(gallery.source(), updates) == Result::SUCCESS);
    REQUIRE(updates.size() == 1);
    CHECK(updates.front().id == "remote");
    CHECK(updates.front().version == "2.0.0");
    CHECK(Find("remote")->version == "1.0.0");

    REQUIRE(Plugin::Gallery::Update(gallery.source(), {"other"}, updated) == Result::SUCCESS);
    CHECK(updated.empty());
    CHECK(Find("remote")->version == "1.0.0");

    REQUIRE(Plugin::Gallery::Update(gallery.source(), {"remote"}, updated) == Result::SUCCESS);
    CHECK(updated == std::vector<std::string>{"remote"});
    plugin = Find("remote");
    REQUIRE(plugin.has_value());
    CHECK(plugin->version == "2.0.0");

    SECTION("installs built for an older CyberEther wait and then update") {
        const auto manifest = Platform::PathFromUtf8(plugin->path) / "manifest.yml";
        auto spec2 = spec;
        spec2.cyberether = "0.0.1";
        {
            std::ofstream file(manifest, std::ios::binary | std::ios::trunc);
            file << Manifest(spec2);
        }
        plugin = Find("remote");
        REQUIRE(plugin.has_value());
        CHECK(plugin->status == Plugin::Status::Waiting);

        REQUIRE(Plugin::Gallery::Update(gallery.source(), {"remote"}, updated) == Result::SUCCESS);
        CHECK(updated == std::vector<std::string>{"remote"});
        plugin = Find("remote");
        REQUIRE(plugin.has_value());
        CHECK(plugin->status == Plugin::Status::Restart);
    }

    SECTION("file installs stop gallery updates") {
        const auto local = temp.root / "local.cep";
        WriteGarbageBundle(local, spec);
        std::string id;
        REQUIRE(Plugin::Install(Platform::PathToUtf8(local), id) == Result::SUCCESS);
        plugin = Find("remote");
        REQUIRE(plugin.has_value());
        CHECK_FALSE(plugin->gallery);

        gallery.requests.clear();
        REQUIRE(Plugin::Gallery::Update(gallery.source(), {"remote"}, updated) == Result::SUCCESS);
        CHECK(updated.empty());
        CHECK(gallery.requests.empty());
    }

    SECTION("bundles that do not match the listing are rejected") {
        gallery.list(Listing({Entry("remote", "3.0.0", "https://files.example/remote-2.cep", Sha256(second),
                                    second.size())}),
                     query);
        CHECK(Plugin::Gallery::Install(gallery.source(), "remote") == Result::ERROR);
        CHECK(JST_LOG_LAST_ERROR().find("but the gallery listed") != std::string::npos);
        plugin = Find("remote");
        REQUIRE(plugin.has_value());
        CHECK(plugin->version == "2.0.0");
    }

    SECTION("unknown and waiting plugins are reported") {
        gallery.list(Listing({}), Plugin::Gallery::Current({"nothere"}));
        CHECK(Plugin::Gallery::Install(gallery.source(), "nothere") == Result::ERROR);
        CHECK(JST_LOG_LAST_ERROR().find("is not in the CyberEther gallery") != std::string::npos);

        gallery.list(Listing({"{\"id\":\"pending\"}"}), Plugin::Gallery::Current({"pending"}));
        CHECK(Plugin::Gallery::Install(gallery.source(), "pending") == Result::ERROR);
        CHECK(JST_LOG_LAST_ERROR().find("does not have a build for CyberEther") != std::string::npos);

        CHECK(Plugin::Gallery::Install(gallery.source(), "https://example.org/x") == Result::ERROR);
    }

    REQUIRE(Plugin::Remove("remote") == Result::SUCCESS);
}

#endif

#endif
