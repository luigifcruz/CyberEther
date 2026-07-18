#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include <zlib.h>

#include "jetstream/logger.hh"
#include "jetstream/platform.hh"
#include "jetstream/plugin.hh"

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

#if defined(_WIN32)
class EnvironmentGuard {
 public:
    explicit EnvironmentGuard(const wchar_t* name) : name_(name) {
        if (const wchar_t* value = _wgetenv(name)) {
            previous_ = value;
        }
    }

    ~EnvironmentGuard() {
        (void)_wputenv_s(name_.c_str(), previous_ ? previous_->c_str() : L"");
    }

    bool set(const std::wstring& value) const {
        return _wputenv_s(name_.c_str(), value.c_str()) == 0;
    }

 private:
    std::wstring name_;
    std::optional<std::wstring> previous_;
};
#else
class EnvironmentGuard {
 public:
    explicit EnvironmentGuard(const char* name) : name_(name) {
        if (const char* value = std::getenv(name)) {
            previous_ = value;
        }
    }

    ~EnvironmentGuard() {
        if (previous_) {
            (void)setenv(name_.c_str(), previous_->c_str(), 1);
        } else {
            (void)unsetenv(name_.c_str());
        }
    }

    bool set(const std::string& value) const {
        return setenv(name_.c_str(), value.c_str(), 1) == 0;
    }

 private:
    std::string name_;
    std::optional<std::string> previous_;
};
#endif

class PluginCacheSandbox {
 public:
    PluginCacheSandbox()
        : directory_("cache"),
#if defined(_WIN32)
          environment_(L"LOCALAPPDATA")
#elif defined(__APPLE__)
          environment_("CFFIXED_USER_HOME")
#else
          environment_("XDG_CACHE_HOME")
#endif
    {
#if defined(_WIN32)
        const bool configured = environment_.set(directory_.root.wstring());
#else
        const bool configured = environment_.set(directory_.root.string());
#endif
        if (!configured) {
            throw std::runtime_error("failed to configure plugin cache sandbox");
        }

        std::string cachePath;
        if (Platform::CachePath(cachePath) != Result::SUCCESS) {
            throw std::runtime_error("failed to resolve plugin cache sandbox");
        }

        std::error_code ec;
        const auto relative = std::filesystem::relative(
            Platform::PathFromUtf8(cachePath), directory_.root, ec);
        if (ec || relative.empty() || relative.is_absolute() ||
            *relative.begin() == "..") {
            throw std::runtime_error("plugin cache escaped its test sandbox");
        }
    }

 private:
    TempDirectory directory_;
    EnvironmentGuard environment_;
};

void EnsurePluginCacheSandbox() {
    static const PluginCacheSandbox sandbox;
    (void)sandbox;
}

std::size_t CountExtractionDirectories() {
    EnsurePluginCacheSandbox();

    std::string cachePath;
    if (Platform::CachePath(cachePath) != Result::SUCCESS) {
        throw std::runtime_error("failed to resolve plugin cache path");
    }

    const auto root = Platform::PathFromUtf8(cachePath) / "registry-plugins";
    std::error_code ec;
    if (!std::filesystem::exists(root, ec)) {
        if (ec) {
            throw std::runtime_error("failed to inspect plugin cache path");
        }
        return 0;
    }

    std::size_t count = 0;
    for (std::filesystem::recursive_directory_iterator entry(root, ec), end;
         entry != end && !ec;
         entry.increment(ec)) {
        if (entry->is_directory(ec) &&
            Platform::PathToUtf8(entry->path().filename()).ends_with(".contents")) {
            ++count;
        }
    }
    if (ec) {
        throw std::runtime_error("failed to inspect plugin extraction directories");
    }
    return count;
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

std::vector<std::uint8_t> Gzip(const std::vector<std::uint8_t>& input) {
    if (input.size() > static_cast<std::size_t>(std::numeric_limits<uInt>::max())) {
        throw std::runtime_error("plugin test payload is too large");
    }

    z_stream stream = {};
    if (deflateInit2(&stream,
                     Z_DEFAULT_COMPRESSION,
                     Z_DEFLATED,
                     15 + 16,
                     8,
                     Z_DEFAULT_STRATEGY) != Z_OK) {
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

void WriteOctal(std::uint8_t* field, std::size_t width, std::uint64_t value) {
    std::ostringstream encoded;
    encoded << std::oct << std::setfill('0') << std::setw(static_cast<int>(width - 1)) << value;
    const auto text = encoded.str();
    if (text.size() > width - 1) {
        throw std::runtime_error("plugin test tar field overflow");
    }

    std::fill(field, field + width, 0);
    std::copy(text.begin(), text.end(), field);
}

std::vector<std::uint8_t> TarHeader(const std::string& name,
                                    std::uint64_t size,
                                    char type = '0') {
    if (name.empty() || name.size() >= 100) {
        throw std::runtime_error("invalid plugin test tar path");
    }

    std::vector<std::uint8_t> header(512, 0);
    std::copy(name.begin(), name.end(), header.begin());
    WriteOctal(header.data() + 124, 12, size);
    header[156] = static_cast<std::uint8_t>(type);
    std::fill(header.begin() + 148, header.begin() + 156, static_cast<std::uint8_t>(' '));

    std::uint64_t checksum = 0;
    for (const auto byte : header) {
        checksum += byte;
    }
    WriteOctal(header.data() + 148, 8, checksum);
    return header;
}

struct TarEntry {
    std::string name;
    std::string content;
    char type = '0';
};

std::vector<std::uint8_t> TarArchive(const std::vector<TarEntry>& entries,
                                     std::size_t terminatorBlocks = 2) {
    std::vector<std::uint8_t> archive;
    for (const auto& entry : entries) {
        auto header = TarHeader(entry.name, entry.content.size(), entry.type);
        archive.insert(archive.end(), header.begin(), header.end());
        archive.insert(archive.end(), entry.content.begin(), entry.content.end());

        const auto padding = (512 - (entry.content.size() % 512)) % 512;
        archive.insert(archive.end(), padding, 0);
    }
    archive.insert(archive.end(), terminatorBlocks * 512, 0);
    return archive;
}

struct ManifestSpec {
    std::string name = "fixture";
    std::string version = "1.0.0";
    std::string minimumVersion = "0.0.0";
    std::string targetPath = "target.bin";
    std::string targetSystem = "unsupported-system";
    std::string targetDevice = "cpu";
    std::string targetArch = "unsupported-arch";
    bool includeTargets = true;
    std::vector<std::string> examples;
};

std::string Manifest(const ManifestSpec& spec) {
    std::string manifest =
        "metadata:\n"
        "  name: \"" + spec.name + "\"\n"
        "  version: \"" + spec.version + "\"\n"
        "  minimumJetstreamVersion: \"" + spec.minimumVersion + "\"\n";

    if (spec.includeTargets) {
        manifest +=
            "targets:\n"
            "  - path: \"" + spec.targetPath + "\"\n"
            "    system: \"" + spec.targetSystem + "\"\n"
            "    device: \"" + spec.targetDevice + "\"\n"
            "    arch: \"" + spec.targetArch + "\"\n";
    } else {
        manifest += "targets: []\n";
    }

    if (spec.examples.empty()) {
        manifest += "examples: []\n";
    } else {
        manifest += "examples:\n";
        for (const auto& path : spec.examples) {
            manifest += "  - path: \"" + path + "\"\n";
        }
    }
    return manifest;
}

void WriteBundle(const std::filesystem::path& path,
                 const ManifestSpec& spec,
                 std::vector<TarEntry> additionalEntries = {},
                 std::size_t terminatorBlocks = 2) {
    std::vector<TarEntry> entries = {{"manifest.yml", Manifest(spec)}};
    entries.insert(entries.end(), additionalEntries.begin(), additionalEntries.end());
    WriteBytes(path, Gzip(TarArchive(entries, terminatorBlocks)));
}

void RequirePluginRejected(const std::string& path, bool reload = false) {
    EnsurePluginCacheSandbox();
    const auto count = Plugin::List().size();
    JST_LOG_LAST_ERROR().clear();

    const auto result = reload ? Plugin::Reload(path) : Plugin::Load(path);
    REQUIRE(result == Result::ERROR);
    REQUIRE(Plugin::List().size() == count);
}

std::string CurrentSystem() {
#if defined(JST_OS_MAC)
    return "macos";
#elif defined(JST_OS_LINUX)
    return "linux";
#elif defined(JST_OS_WINDOWS)
    return "windows";
#elif defined(JST_OS_ANDROID)
    return "android";
#elif defined(JST_OS_IOS)
    return "ios";
#elif defined(JST_OS_BROWSER)
    return "browser";
#else
    return "unknown";
#endif
}

std::string CurrentArch() {
#if defined(__aarch64__) || defined(_M_ARM64)
    return "arm64";
#elif defined(__x86_64__) || defined(_M_X64)
    return "x86_64";
#elif defined(__i386__) || defined(_M_IX86)
    return "x86";
#elif defined(__arm__) || defined(_M_ARM)
    return "arm";
#elif defined(__wasm32__)
    return "wasm32";
#else
    return "unknown";
#endif
}

ManifestSpec CompatibleManifest() {
    ManifestSpec spec;
    spec.targetSystem = CurrentSystem();
    spec.targetArch = CurrentArch();
    return spec;
}

}  // namespace

TEST_CASE("Plugin ABI metadata remains a fixed public contract",
          "[core][extensions][plugin]") {
    const JetstreamPluginAbi abi = {
        JETSTREAM_PLUGIN_ABI_MAGIC,
        static_cast<std::uint32_t>(sizeof(JetstreamPluginAbi)),
        JETSTREAM_PLUGIN_ABI_VERSION,
    };

    REQUIRE(sizeof(JetstreamPluginAbi) == 3 * sizeof(std::uint32_t));
    REQUIRE(abi.magic == UINT32_C(0x4a535450));
    REQUIRE(abi.size == sizeof(JetstreamPluginAbi));
    REQUIRE(abi.abi_version == 1);
}

TEST_CASE("Plugin public entry points reject invalid bundle paths",
          "[core][extensions][plugin]") {
    SECTION("empty paths") {
        RequirePluginRejected("");
        RequirePluginRejected("", true);
    }

    SECTION("non-bundle extensions") {
        RequirePluginRejected("cyberether-extension.so");
        RequirePluginRejected("cyberether-extension.tar.gz", true);
    }

    SECTION("missing bundles") {
        const TempDirectory temp("missing");
        RequirePluginRejected(Platform::PathToUtf8(temp.root / "missing.cep"));
    }

    SECTION("bundle paths that are directories") {
        const TempDirectory temp("directory");
        const auto directory = temp.root / "directory.cep";
        std::error_code ec;
        REQUIRE(std::filesystem::create_directory(directory, ec));
        REQUIRE_FALSE(ec);
        RequirePluginRejected(Platform::PathToUtf8(directory));
    }
}

TEST_CASE("Plugin loader rejects malformed CEP compression and tar structure",
          "[core][extensions][plugin]") {
    const TempDirectory temp("archive");
    const auto archivePath = temp.root / "fixture.cep";
    const auto archivePathUtf8 = Platform::PathToUtf8(archivePath);

    SECTION("empty file") {
        WriteBytes(archivePath, {});
        RequirePluginRejected(archivePathUtf8);
    }

    SECTION("non-gzip bytes") {
        WriteBytes(archivePath, {'n', 'o', 't', '-', 'g', 'z', 'i', 'p'});
        RequirePluginRejected(archivePathUtf8);
    }

    SECTION("truncated gzip stream") {
        auto archive = Gzip(TarArchive({{"manifest.yml", "metadata: {}\n"}}));
        archive.resize(archive.size() - 4);
        WriteBytes(archivePath, archive);
        RequirePluginRejected(archivePathUtf8);
    }

    SECTION("gzip with a partial tar header") {
        WriteBytes(archivePath, Gzip({'t', 'a', 'r'}));
        RequirePluginRejected(archivePathUtf8);
    }

    SECTION("tar entry extending beyond archive bounds") {
        WriteBytes(archivePath, Gzip(TarHeader("manifest.yml", 32)));
        RequirePluginRejected(archivePathUtf8);
    }

    SECTION("tar checksum mismatch") {
        auto archive = TarArchive({{"manifest.yml", "metadata: {}\n"}});
        archive[0] ^= 0x01;
        WriteBytes(archivePath, Gzip(archive));
        RequirePluginRejected(archivePathUtf8);
    }

    SECTION("tar without an end marker") {
        WriteBytes(archivePath,
                   Gzip(TarArchive({{"manifest.yml", "metadata: {}\n"}}, 0)));
        RequirePluginRejected(archivePathUtf8);
    }

    SECTION("unsafe archive path") {
        WriteBytes(archivePath,
                   Gzip(TarArchive({{"../manifest.yml", "metadata: {}\n"}})));
        RequirePluginRejected(archivePathUtf8);
    }

    SECTION("unsupported archive entry type") {
        WriteBytes(archivePath,
                   Gzip(TarArchive({{"manifest.yml", "", '2'}})));
        RequirePluginRejected(archivePathUtf8);
    }

    SECTION("empty tar") {
        WriteBytes(archivePath, Gzip(std::vector<std::uint8_t>(1024, 0)));
        RequirePluginRejected(archivePathUtf8);
    }

    SECTION("reload rejects the same malformed input") {
        WriteBytes(archivePath, {'b', 'a', 'd'});
        RequirePluginRejected(archivePathUtf8, true);
    }

    SECTION("single-block tar terminator") {
        const auto extractionCount = CountExtractionDirectories();
        WriteBundle(archivePath, ManifestSpec{}, {}, 1);
        RequirePluginRejected(archivePathUtf8);
        REQUIRE(JST_LOG_LAST_ERROR().find("missing an end marker") != std::string::npos);
        REQUIRE(CountExtractionDirectories() == extractionCount);
    }

    SECTION("partial second tar terminator block") {
        auto archive = TarArchive({{"manifest.yml", "metadata: {}\n"}}, 1);
        archive.insert(archive.end(), 511, 0);
        WriteBytes(archivePath, Gzip(archive));
        RequirePluginRejected(archivePathUtf8);
        REQUIRE(JST_LOG_LAST_ERROR().find("missing an end marker") != std::string::npos);
    }
}

TEST_CASE("Plugin manifests are rejected before target loading",
          "[core][extensions][plugin]") {
    const TempDirectory temp("manifest");
    const auto archivePath = temp.root / "fixture.cep";
    const auto archivePathUtf8 = Platform::PathToUtf8(archivePath);

    SECTION("missing manifest") {
        WriteBytes(archivePath, Gzip(TarArchive({{"other.yml", "metadata: {}\n"}})));
        RequirePluginRejected(archivePathUtf8);
    }

    SECTION("malformed YAML") {
        WriteBytes(archivePath, Gzip(TarArchive({{"manifest.yml", "metadata: [\n"}})));
        RequirePluginRejected(archivePathUtf8);
    }

    SECTION("incomplete metadata") {
        auto spec = ManifestSpec{};
        spec.name.clear();
        WriteBundle(archivePath, spec);
        RequirePluginRejected(archivePathUtf8);
    }

    SECTION("invalid minimum version") {
        auto spec = ManifestSpec{};
        spec.minimumVersion = "1.2";
        WriteBundle(archivePath, spec);
        RequirePluginRejected(archivePathUtf8);
    }

    SECTION("minimum version newer than the host") {
        auto spec = ManifestSpec{};
        spec.minimumVersion = "255.255.255";
        WriteBundle(archivePath, spec);
        RequirePluginRejected(archivePathUtf8);
    }

    SECTION("missing targets") {
        auto spec = ManifestSpec{};
        spec.includeTargets = false;
        WriteBundle(archivePath, spec);
        RequirePluginRejected(archivePathUtf8);
    }

    SECTION("unsafe relative target path") {
        auto spec = ManifestSpec{};
        spec.targetPath = "../target.bin";
        WriteBundle(archivePath, spec);
        RequirePluginRejected(archivePathUtf8);
    }

    SECTION("absolute target path") {
        auto spec = ManifestSpec{};
        spec.targetPath = "/target.bin";
        WriteBundle(archivePath, spec);
        RequirePluginRejected(archivePathUtf8);
    }

    SECTION("incomplete target") {
        auto spec = ManifestSpec{};
        spec.targetSystem.clear();
        WriteBundle(archivePath, spec);
        RequirePluginRejected(archivePathUtf8);
    }

    SECTION("unknown target device") {
        auto spec = ManifestSpec{};
        spec.targetDevice = "not-a-device";
        WriteBundle(archivePath, spec);
        RequirePluginRejected(archivePathUtf8);
    }

    SECTION("unsafe example path") {
        auto spec = ManifestSpec{};
        spec.examples = {"../example.yml"};
        WriteBundle(archivePath, spec);
        RequirePluginRejected(archivePathUtf8);
    }

    SECTION("malformed plugin version") {
        auto spec = ManifestSpec{};
        spec.version = "not-a-version";
        WriteBundle(archivePath, spec);
        RequirePluginRejected(archivePathUtf8);
        // Current defect: metadata.version syntax is not validated before target selection.
        CHECK((JST_LOG_LAST_ERROR().find("invalid") != std::string::npos &&
               JST_LOG_LAST_ERROR().find("version") != std::string::npos));
    }

    SECTION("well-formed bundle without a compatible target") {
        WriteBundle(archivePath, ManifestSpec{});
        RequirePluginRejected(archivePathUtf8);
    }
}

TEST_CASE("Plugin loader rejects compatible targets without loadable artifacts",
          "[core][extensions][plugin]") {
    const TempDirectory temp("loader");
    const auto archivePath = temp.root / "fixture.cep";
    const auto archivePathUtf8 = Platform::PathToUtf8(archivePath);

    SECTION("target is absent from the archive") {
        WriteBundle(archivePath, CompatibleManifest());
        RequirePluginRejected(archivePathUtf8);
        REQUIRE(JST_LOG_LAST_ERROR().find("Failed to load plugin") != std::string::npos);
    }

    SECTION("target is not a dynamic library") {
        WriteBundle(archivePath,
                    CompatibleManifest(),
                    {{"target.bin", "not a dynamic library"}});
        RequirePluginRejected(archivePathUtf8);
        REQUIRE(JST_LOG_LAST_ERROR().find("Failed to load plugin") != std::string::npos);
    }
}
