#include "jetstream/plugin.hh"

#include "plugin_gallery.hh"
#include "tools/text.hh"

#include "jetstream/home.hh"
#include "jetstream/logger.hh"
#include "jetstream/platform.hh"
#include "jetstream/parser.hh"
#include "jetstream/registry.hh"
#include "jetstream/runtime.hh"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <miniz.h>

namespace Jetstream {

namespace {

constexpr const char* ManifestFile = "manifest.yml";
constexpr const char* IndexFile = "index.yml";

constexpr uint64_t MaxBundleSize = 512ull << 20;
constexpr uint64_t MaxUncompressedSize = 2ull << 30;
constexpr uint64_t MaxBundleEntries = 10000;
constexpr uint64_t MaxManifestSize = 64ull << 10;

struct InstallOptions {
    bool gallery = false;
    bool update = false;
    std::string id;
    std::string version;
};

std::string ReadText(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw Result::ERROR;
    }
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

}  // namespace

struct Plugin::Impl {
    struct Manifest {
        std::string id;
        std::string title;
        std::string version;
        std::string cyberether;
        std::string build;
        std::string summary;
        std::string description;
        std::string author;
        std::string license;

        JST_SERDES(id, title, version, cyberether, build, summary, description, author, license);
    };

    struct IndexEntry {
        std::string id;
        std::string folder;
        std::string pending;
        bool gallery = false;

        JST_SERDES(id, folder, pending, gallery);
    };

    struct Index {
        std::vector<IndexEntry> plugins;

        JST_SERDES(plugins);
    };

    struct Registrations {
        struct Module {
            std::string type;
            DeviceType device = DeviceType::None;
            RuntimeType runtime = RuntimeType::NONE;
            ProviderType provider;
        };

        struct Benchmark {
            std::string moduleType;
            const void* owner = nullptr;
        };

        std::vector<Module> modules;
        std::vector<std::string> blocks;
        std::vector<Benchmark> benchmarks;
    };

    struct Record {
        Manifest manifest;
        std::filesystem::path root;
        std::string source;
        bool temporary = false;
        void* handle = nullptr;
        Registrations registrations;
        std::vector<std::string> examples;
        Platform::FileLock user;
    };

    Impl();
    ~Impl();

    Result install(const std::string& path, std::string& id, const InstallOptions& options = {});
    Result installEntry(const Gallery::Source& source,
                        const Gallery::Entry& entry,
                        bool update,
                        const std::function<bool()>& cancelled);
    Result remove(const std::string& id);
    Result loadInstalled();
    Result load(const std::string& path);
    Result reload(const std::string& id);
    std::vector<Plugin::Info> list();
    Result collect(std::vector<Plugin::Info>& rows);
    std::filesystem::path path(const std::string& id);
    std::string owner(const std::string& block, const std::string& provider);

 private:
    static bool validName(const std::string& value, U64 limit);
    static std::string galleryOrigin(const std::string& id);
    static std::string libraryExtension();
    static std::filesystem::path libraryPath(const std::filesystem::path& root, const std::string& id);
    static bool safeRelativePath(const std::string& rawPath, std::filesystem::path& relativePath);
    static std::string uniqueStamp();

    static Result extract(const std::filesystem::path& source, const std::filesystem::path& destination);
    static Result readManifest(const std::filesystem::path& root, Manifest& manifest);
    static Result validateManifest(const std::filesystem::path& root, const Manifest& manifest);
    static bool compatible(const std::filesystem::path& root, const Manifest& manifest);

    static Registrations snapshotRegistry();
    static Registrations diffRegistrations(const Registrations& before, const Registrations& after);
    static void rollbackRegistrations(const Registrations& registrations);
    static Result validatePrefixes(const std::string& id,
                                   const Registrations& registrations,
                                   const Registrations& before);
    static Result publishExamples(const Record& plugin, std::vector<std::string>& keys);

    static Result storeRoot(std::filesystem::path& root, bool create = true);
    static Result readIndex(const std::filesystem::path& root, Index& index);
    static Result writeIndex(const std::filesystem::path& root, const Index& index);
    static void sweepStore(const std::filesystem::path& root, const Index& index);
    static bool folderInUse(const std::filesystem::path& folder);
    static void pruneUsers(const std::filesystem::path& folder);

    Result cacheDirectory(std::filesystem::path& directory);
    static void sweepCache(const std::filesystem::path& runs);
    void cleanupCache();

    Result open(Record& plugin);
    void close(Record& plugin);
    bool loaded(const std::string& id);
    void cleanup();

    std::mutex lifecycleMutex;
    std::mutex pluginsMutex;
    std::mutex pathsMutex;
    std::mutex cacheMutex;

    std::vector<Record> plugins;
    std::map<std::string, std::filesystem::path> paths;
    std::set<std::string> failed;

    Platform::FileLock cacheOwnerLock;
    std::filesystem::path cacheRunDirectory;
    uint64_t cacheGeneration = 0;
};

Plugin::Impl::Impl() {
    (void)Home::Get();
}

Plugin::Impl::~Impl() {
    cleanup();
}

bool Plugin::Impl::validName(const std::string& value, U64 limit) {
    if (value.empty() || value.size() > limit ||
        !std::isalnum(static_cast<unsigned char>(value.front())) ||
        value.find("..") != std::string::npos) {
        return false;
    }

    return std::all_of(value.begin(), value.end(), [](unsigned char character) {
        return std::isalnum(character) || character == '.' || character == '-' || character == '+';
    });
}

std::string Plugin::Impl::galleryOrigin(const std::string& id) {
    return "plugin/" + id;
}

std::string Plugin::Impl::libraryExtension() {
#if defined(JST_OS_MAC) || defined(JST_OS_IOS)
    return ".dylib";
#elif defined(JST_OS_WINDOWS)
    return ".dll";
#elif defined(JST_OS_BROWSER)
    return ".wasm";
#else
    return ".so";
#endif
}

std::filesystem::path Plugin::Impl::libraryPath(const std::filesystem::path& root, const std::string& id) {
    return root / "lib" / Platform::PathFromUtf8(id + libraryExtension());
}

bool Plugin::Impl::safeRelativePath(const std::string& rawPath, std::filesystem::path& relativePath) {
    if (rawPath.find_first_of(std::string_view("\\:\0", 3)) != std::string::npos) {
        return false;
    }

    std::size_t start = 0;
    while (start <= rawPath.size()) {
        const auto end = std::min(rawPath.find('/', start), rawPath.size());
        const auto segment = std::string_view(rawPath).substr(start, end - start);
        if (segment.empty() || segment.back() == '.' || segment.back() == ' ' || Gallery::ReservedName(segment)) {
            return false;
        }
        start = end + 1;
    }

    try {
        relativePath = Platform::PathFromUtf8(rawPath);
    } catch (...) {
        return false;
    }
    return true;
}

std::string Plugin::Impl::uniqueStamp() {
    static std::atomic<uint64_t> sequence = 0;
    const auto timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    return std::to_string(timestamp) + "-" + std::to_string(sequence.fetch_add(1));
}

Result Plugin::Impl::extract(const std::filesystem::path& source, const std::filesystem::path& destination) {
    const auto sourceName = Platform::PathToUtf8(source);

    std::error_code ec;
    const uint64_t size = std::filesystem::file_size(source, ec);

    std::ifstream file;
    if (!ec) {
        file.open(source, std::ios::binary);
    }
    if (ec || !file) {
        JST_ERROR("[PLUGIN] Failed to open '{}'.", sourceName);
        return Result::ERROR;
    }

    if (size > MaxBundleSize) {
        JST_ERROR("[PLUGIN] Plugin bundle '{}' is larger than 512 MiB.", sourceName);
        return Result::ERROR;
    }

    mz_zip_archive zip = {};
    zip.m_pIO_opaque = &file;
    zip.m_pRead = [](void* opaque, mz_uint64 offset, void* data, size_t count) -> size_t {
        auto& stream = *static_cast<std::ifstream*>(opaque);
        stream.clear();
        stream.seekg(static_cast<std::streamoff>(offset));
        stream.read(static_cast<char*>(data), static_cast<std::streamsize>(count));
        return static_cast<size_t>(stream.gcount());
    };

    auto zipError = [&]() {
        return mz_zip_get_error_string(mz_zip_peek_last_error(&zip));
    };

    if (!mz_zip_reader_init(&zip, size, 0)) {
        JST_ERROR("[PLUGIN] Failed to read plugin bundle '{}': {}.", sourceName, zipError());
        return Result::ERROR;
    }

    struct Reader {
        mz_zip_archive& zip;
        ~Reader() {
            mz_zip_reader_end(&zip);
        }
    } reader{zip};

    struct Entry {
        mz_uint index = 0;
        std::filesystem::path path;
        bool directory = false;
        uint32_t mode = 0;
        uint64_t size = 0;
    };

    auto allowed = [](const std::filesystem::path& path) {
        const auto top = *path.begin();
        return path == ManifestFile || top == "lib" || top == "data" || top == "examples";
    };

    const auto count = mz_zip_reader_get_num_files(&zip);
    if (count > MaxBundleEntries) {
        JST_ERROR("[PLUGIN] Plugin bundle '{}' has more than {} entries.", sourceName, MaxBundleEntries);
        return Result::ERROR;
    }

    std::vector<Entry> entries;
    std::set<std::string> seen;
    uint64_t totalSize = 0;

    for (mz_uint index = 0; index < count; ++index) {
        mz_zip_archive_file_stat stat;
        if (!mz_zip_reader_file_stat(&zip, index, &stat)) {
            JST_ERROR("[PLUGIN] Failed to read plugin bundle '{}': {}.", sourceName, zipError());
            return Result::ERROR;
        }

        std::string name(mz_zip_reader_get_filename(&zip, index, nullptr, 0), '\0');
        mz_zip_reader_get_filename(&zip, index, name.data(), static_cast<mz_uint>(name.size()));
        name.pop_back();

        Entry entry;
        entry.index = index;
        entry.directory = !name.empty() && name.back() == '/';
        entry.size = stat.m_uncomp_size;
        if ((stat.m_version_made_by >> 8) == 3) {
            entry.mode = stat.m_external_attr >> 16;
        }

        const auto trimmed = entry.directory ? name.substr(0, name.size() - 1) : name;
        if (!safeRelativePath(trimmed, entry.path)) {
            JST_ERROR("[PLUGIN] Unsafe bundle entry path '{}'.", name);
            return Result::ERROR;
        }

        if (!allowed(entry.path)) {
            JST_ERROR("[PLUGIN] Unexpected bundle entry '{}'. Bundles may only contain manifest.yml, lib/, data/, "
                      "and examples/.", name);
            return Result::ERROR;
        }

        if (!seen.insert(Text::ToLower(trimmed)).second) {
            JST_ERROR("[PLUGIN] Duplicate bundle entry '{}'.", name);
            return Result::ERROR;
        }

        if (entry.path == ManifestFile && entry.size > MaxManifestSize) {
            JST_ERROR("[PLUGIN] Plugin bundle '{}' has a manifest.yml larger than 64 KiB.", sourceName);
            return Result::ERROR;
        }

        if (entry.size > MaxUncompressedSize - totalSize) {
            JST_ERROR("[PLUGIN] Plugin bundle '{}' expands to more than 2 GiB.", sourceName);
            return Result::ERROR;
        }
        totalSize += entry.size;

        entries.push_back(std::move(entry));
    }

    std::filesystem::create_directories(destination, ec);
    if (ec) {
        JST_ERROR("[PLUGIN] Failed to create plugin extraction directory '{}'.",
                  Platform::PathToUtf8(destination));
        return Result::ERROR;
    }

    struct Sink {
        std::ofstream output;
        uint64_t remaining = 0;
    };

    const mz_file_write_func write = [](void* opaque, mz_uint64, const void* data, size_t count) -> size_t {
        auto& sink = *static_cast<Sink*>(opaque);
        if (count > sink.remaining) {
            return 0;
        }
        sink.output.write(static_cast<const char*>(data), static_cast<std::streamsize>(count));
        if (!sink.output) {
            return 0;
        }
        sink.remaining -= count;
        return count;
    };

    for (const auto& entry : entries) {
        const auto outputPath = destination / entry.path;
        if (entry.directory) {
            std::filesystem::create_directories(outputPath, ec);
            if (ec) {
                JST_ERROR("[PLUGIN] Failed to create directory '{}'.", Platform::PathToUtf8(outputPath));
                return Result::ERROR;
            }
            continue;
        }

        std::filesystem::create_directories(outputPath.parent_path(), ec);
        if (ec) {
            JST_ERROR("[PLUGIN] Failed to create directory '{}'.",
                      Platform::PathToUtf8(outputPath.parent_path()));
            return Result::ERROR;
        }

        Sink sink;
        sink.remaining = entry.size;
        sink.output.open(outputPath, std::ios::binary | std::ios::trunc);
        if (!sink.output) {
            JST_ERROR("[PLUGIN] Failed to create file '{}'.", Platform::PathToUtf8(outputPath));
            return Result::ERROR;
        }

        if (!mz_zip_reader_extract_to_callback(&zip, entry.index, write, &sink, 0)) {
            JST_ERROR("[PLUGIN] Failed to extract bundle entry '{}': {}.",
                      Platform::PathToUtf8(entry.path),
                      zipError());
            return Result::ERROR;
        }

        if (sink.remaining != 0) {
            JST_ERROR("[PLUGIN] Bundle entry '{}' is smaller than its declared size.",
                      Platform::PathToUtf8(entry.path));
            return Result::ERROR;
        }

        sink.output.close();
        if (!sink.output) {
            JST_ERROR("[PLUGIN] Failed to write file '{}'.", Platform::PathToUtf8(outputPath));
            return Result::ERROR;
        }

#if !defined(JST_OS_WINDOWS)
        if ((entry.mode & 0111) != 0) {
            std::filesystem::permissions(outputPath,
                                         static_cast<std::filesystem::perms>(entry.mode & 0111),
                                         std::filesystem::perm_options::add,
                                         ec);
            if (ec) {
                JST_ERROR("[PLUGIN] Failed to mark '{}' as executable.", Platform::PathToUtf8(outputPath));
                return Result::ERROR;
            }
        }
#endif
    }

    return Result::SUCCESS;
}

Result Plugin::Impl::readManifest(const std::filesystem::path& root, Manifest& manifest) {
    manifest = {};

    std::string content;
    try {
        content = ReadText(root / ManifestFile);
    } catch (...) {
        JST_ERROR("[PLUGIN] Plugin bundle does not contain a manifest.");
        return Result::ERROR;
    }

    Parser::Map data;
    JST_CHECK(Parser::YamlDecode(content, data));

    static const std::set<std::string> keys = {
        "id", "title", "version", "cyberether", "build", "summary", "description", "author", "license",
    };
    for (const auto& entry : data) {
        if (!keys.contains(entry.key)) {
            JST_ERROR("[PLUGIN] Plugin manifest has an unknown key '{}'.", entry.key);
            return Result::ERROR;
        }
    }

    JST_CHECK(manifest.deserialize(data));
    return Result::SUCCESS;
}

Result Plugin::Impl::validateManifest(const std::filesystem::path& root, const Manifest& manifest) {
    if (!Gallery::ValidId(manifest.id)) {
        JST_ERROR("[PLUGIN] Plugin ID '{}' is invalid. Use 1 to 32 lowercase letters and digits, "
                  "starting with a letter, and avoid names reserved by Windows.", manifest.id);
        return Result::ERROR;
    }

    if (manifest.title.empty()) {
        JST_ERROR("[PLUGIN] Plugin '{}' does not have a title.", manifest.id);
        return Result::ERROR;
    }

    if (!validName(manifest.version, 64)) {
        JST_ERROR("[PLUGIN] Plugin '{}' has an invalid version '{}'.", manifest.id, manifest.version);
        return Result::ERROR;
    }

    if (manifest.cyberether != JETSTREAM_VERSION_STR) {
        JST_ERROR("[PLUGIN] Plugin '{}' was built for CyberEther {}, but this is CyberEther {}.",
                  manifest.id,
                  manifest.cyberether.empty() ? "an unknown version" : manifest.cyberether,
                  JETSTREAM_VERSION_STR);
        return Result::ERROR;
    }

    if (manifest.build != JETSTREAM_BUILD_ID) {
        JST_ERROR("[PLUGIN] Plugin '{}' was built for {}, but this CyberEther is {}.",
                  manifest.id,
                  manifest.build.empty() ? "an unknown build" : manifest.build,
                  JETSTREAM_BUILD_ID);
        return Result::ERROR;
    }

    std::error_code ec;
    if (!std::filesystem::is_regular_file(libraryPath(root, manifest.id), ec)) {
        JST_ERROR("[PLUGIN] Plugin '{}' does not include lib/{}{}.", manifest.id, manifest.id, libraryExtension());
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

bool Plugin::Impl::compatible(const std::filesystem::path& root, const Manifest& manifest) {
    std::error_code ec;
    return Gallery::ValidId(manifest.id) &&
           manifest.cyberether == JETSTREAM_VERSION_STR &&
           manifest.build == JETSTREAM_BUILD_ID &&
           std::filesystem::is_regular_file(libraryPath(root, manifest.id), ec);
}

Plugin::Impl::Registrations Plugin::Impl::snapshotRegistry() {
    Registrations snapshot;

    for (const auto& module : Registry::ListAvailableModules()) {
        snapshot.modules.push_back({
            .type = module.type,
            .device = module.device,
            .runtime = module.runtime,
            .provider = module.provider,
        });
    }

    for (const auto& block : Registry::ListAvailableBlocks()) {
        snapshot.blocks.push_back(block.type);
    }

    for (const auto& benchmark : Registry::ListAvailableBenchmarks()) {
        snapshot.benchmarks.push_back({
            .moduleType = benchmark.moduleType,
            .owner = benchmark.owner,
        });
    }

    return snapshot;
}

Plugin::Impl::Registrations Plugin::Impl::diffRegistrations(const Registrations& before,
                                                            const Registrations& after) {
    Registrations diff;

    for (const auto& module : after.modules) {
        const auto existing = std::find_if(before.modules.begin(), before.modules.end(), [&](const auto& entry) {
            return entry.type == module.type &&
                   entry.device == module.device &&
                   entry.runtime == module.runtime &&
                   entry.provider == module.provider;
        });
        if (existing == before.modules.end()) {
            diff.modules.push_back(module);
        }
    }

    for (const auto& block : after.blocks) {
        if (std::find(before.blocks.begin(), before.blocks.end(), block) == before.blocks.end()) {
            diff.blocks.push_back(block);
        }
    }

    for (const auto& benchmark : after.benchmarks) {
        const auto existing = std::find_if(before.benchmarks.begin(), before.benchmarks.end(), [&](const auto& entry) {
            return entry.moduleType == benchmark.moduleType && entry.owner == benchmark.owner;
        });
        if (existing == before.benchmarks.end()) {
            diff.benchmarks.push_back(benchmark);
        }
    }

    return diff;
}

void Plugin::Impl::rollbackRegistrations(const Registrations& registrations) {
    for (auto it = registrations.benchmarks.rbegin(); it != registrations.benchmarks.rend(); ++it) {
        (void)Registry::UnregisterBenchmark(it->moduleType, it->owner);
    }

    for (auto it = registrations.blocks.rbegin(); it != registrations.blocks.rend(); ++it) {
        (void)Registry::UnregisterBlock(*it);
    }

    for (auto it = registrations.modules.rbegin(); it != registrations.modules.rend(); ++it) {
        (void)Registry::UnregisterModule(it->type, it->device, it->runtime, it->provider);
    }
}

Result Plugin::Impl::validatePrefixes(const std::string& id,
                                      const Registrations& registrations,
                                      const Registrations& before) {
    const auto prefix = id + "_";

    for (const auto& block : registrations.blocks) {
        if (!block.starts_with(prefix)) {
            JST_ERROR("[PLUGIN] Plugin '{}' registered block '{}' without the '{}' prefix.", id, block, prefix);
            return Result::ERROR;
        }
    }

    for (const auto& module : registrations.modules) {
        const bool existing = std::any_of(before.modules.begin(), before.modules.end(), [&](const auto& entry) {
            return entry.type == module.type;
        });
        const bool owned = module.type.starts_with(prefix) && !existing;
        if (!owned && !module.provider.starts_with(prefix)) {
            JST_ERROR("[PLUGIN] Plugin '{}' registered an implementation of '{}' with provider '{}' "
                      "without the '{}' prefix.",
                      id,
                      module.type,
                      module.provider,
                      prefix);
            return Result::ERROR;
        }
    }

    return Result::SUCCESS;
}

Result Plugin::Impl::publishExamples(const Record& plugin, std::vector<std::string>& keys) {
    keys.clear();

    const auto directory = plugin.root / "examples";
    std::error_code ec;
    if (!std::filesystem::is_directory(directory, ec)) {
        return Result::SUCCESS;
    }

    std::vector<std::filesystem::path> files;
    for (const auto& entry : std::filesystem::directory_iterator(directory, ec)) {
        std::error_code entryEc;
        if (entry.is_regular_file(entryEc) && entry.path().extension() == ".yml") {
            files.push_back(entry.path());
        }
    }
    if (ec) {
        JST_ERROR("[PLUGIN] Failed to list the examples of plugin '{}'.", plugin.manifest.id);
        return Result::ERROR;
    }
    if (files.empty()) {
        return Result::SUCCESS;
    }

    std::sort(files.begin(), files.end());

    Home::Gallery::Section section{
        .key = galleryOrigin(plugin.manifest.id),
        .title = plugin.manifest.title,
        .description = "Examples bundled with this plugin.",
    };

    for (const auto& file : files) {
        std::string content;
        try {
            content = ReadText(file);
        } catch (...) {
            JST_ERROR("[PLUGIN] Failed to read example '{}'.", Platform::PathToUtf8(file));
            return Result::ERROR;
        }

        const auto stem = Platform::PathToUtf8(file.stem());
        Home::Gallery::Flowgraph flowgraph;
        JST_CHECK(Home::Gallery::Flowgraph::Parse(plugin.manifest.id + "/" + stem, std::move(content), flowgraph));
        if (flowgraph.title.empty()) {
            flowgraph.title = stem;
        }
        keys.push_back(flowgraph.key);
        section.flowgraphs.push_back(std::move(flowgraph));
    }

    JST_CHECK(Home::Get().gallery().publish(galleryOrigin(plugin.manifest.id), {std::move(section)}));
    return Result::SUCCESS;
}

Result Plugin::Impl::storeRoot(std::filesystem::path& root, bool create) {
    std::string dataPath;
    JST_CHECK(Platform::DataPath(dataPath));
    root = Platform::PathFromUtf8(dataPath) / "plugins" / JETSTREAM_BUILD_ID;
    if (!create) {
        return Result::SUCCESS;
    }

    std::error_code ec;
    std::filesystem::create_directories(root, ec);
    if (ec) {
        JST_ERROR("[PLUGIN] Failed to create the plugin store '{}'.", Platform::PathToUtf8(root));
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result Plugin::Impl::readIndex(const std::filesystem::path& root, Index& index) {
    index = {};

    std::error_code ec;
    const bool exists = std::filesystem::exists(root / IndexFile, ec);
    if (ec) {
        JST_ERROR("[PLUGIN] Failed to read the plugin store index.");
        return Result::ERROR;
    }
    if (!exists) {
        const bool present = std::filesystem::exists(root, ec);
        if (ec) {
            JST_ERROR("[PLUGIN] Failed to inspect the plugin store '{}'.", Platform::PathToUtf8(root));
            return Result::ERROR;
        }
        if (!present) {
            return Result::SUCCESS;
        }
        for (const auto& entry : std::filesystem::directory_iterator(root, ec)) {
            std::error_code entryEc;
            if (!entry.is_symlink(entryEc) && entry.is_directory(entryEc) &&
                Gallery::ValidId(Platform::PathToUtf8(entry.path().filename()))) {
                JST_ERROR("[PLUGIN] The plugin store index is missing but plugins are still stored in '{}'. "
                          "Restore index.yml or delete that folder to start over.",
                          Platform::PathToUtf8(root));
                return Result::ERROR;
            }
        }
        if (ec) {
            JST_ERROR("[PLUGIN] Failed to inspect the plugin store '{}'.", Platform::PathToUtf8(root));
            return Result::ERROR;
        }
        return Result::SUCCESS;
    }

    std::string content;
    try {
        content = ReadText(root / IndexFile);
    } catch (...) {
        JST_ERROR("[PLUGIN] Failed to read the plugin store index.");
        return Result::ERROR;
    }

    Parser::Map data;
    if (Parser::YamlDecode(content, data) != Result::SUCCESS || !data.contains("plugins") ||
        index.deserialize(data) != Result::SUCCESS) {
        JST_ERROR("[PLUGIN] The plugin store index is corrupted.");
        return Result::ERROR;
    }

    const auto malformed = std::any_of(index.plugins.begin(), index.plugins.end(), [](const IndexEntry& entry) {
        return !Gallery::ValidId(entry.id) || !validName(entry.folder, 160) ||
               (!entry.pending.empty() && !validName(entry.pending, 160));
    });
    std::set<std::string> ids;
    const auto duplicate = std::any_of(index.plugins.begin(), index.plugins.end(), [&](const IndexEntry& entry) {
        return !ids.insert(entry.id).second;
    });
    if (malformed || duplicate) {
        JST_ERROR("[PLUGIN] The plugin store index has a malformed entry.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result Plugin::Impl::writeIndex(const std::filesystem::path& root, const Index& index) {
    std::string content = "plugins: []\n";
    if (!index.plugins.empty()) {
        Parser::Map data;
        JST_CHECK(index.serialize(data));
        JST_CHECK(Parser::YamlEncode(data, content));
    }

    const auto temporary = root / (std::string(IndexFile) + ".tmp");
    {
        std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
        file << content;
        file.close();
        if (!file) {
            JST_ERROR("[PLUGIN] Failed to write the plugin store index.");
            return Result::ERROR;
        }
    }

    std::error_code ec;
    for (int attempt = 0; attempt < 20; ++attempt) {
        ec.clear();
        std::filesystem::rename(temporary, root / IndexFile, ec);
        if (!ec) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
    if (ec) {
        JST_ERROR("[PLUGIN] Failed to replace the plugin store index.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

bool Plugin::Impl::folderInUse(const std::filesystem::path& folder) {
    std::error_code ec;
    const auto users = folder / ".users";
    if (!std::filesystem::is_directory(users, ec)) {
        return false;
    }

    std::filesystem::directory_iterator entries(users, ec);
    for (const auto end = std::filesystem::directory_iterator(); !ec && entries != end; entries.increment(ec)) {
        Platform::FileLock probe;
        if (probe.acquire(Platform::PathToUtf8(entries->path()), false) != Result::SUCCESS) {
            return true;
        }
    }

    return static_cast<bool>(ec);
}

void Plugin::Impl::pruneUsers(const std::filesystem::path& folder) {
    std::error_code ec;
    std::filesystem::directory_iterator entries(folder / ".users", ec);
    for (const auto end = std::filesystem::directory_iterator(); !ec && entries != end; entries.increment(ec)) {
        Platform::FileLock probe;
        if (probe.acquire(Platform::PathToUtf8(entries->path()), false) != Result::SUCCESS) {
            continue;
        }
        probe.release();

        std::error_code removeEc;
        std::filesystem::remove(entries->path(), removeEc);
    }
}

void Plugin::Impl::sweepStore(const std::filesystem::path& root, const Index& index) {
    std::error_code ec;
    std::filesystem::remove_all(root / ".staging", ec);

    if (!std::filesystem::is_regular_file(root / IndexFile, ec)) {
        return;
    }

    for (const auto& plugin : std::filesystem::directory_iterator(root, ec)) {
        std::error_code entryEc;
        if (plugin.is_symlink(entryEc) || !plugin.is_directory(entryEc)) {
            continue;
        }

        const auto id = Platform::PathToUtf8(plugin.path().filename());
        if (!Gallery::ValidId(id)) {
            continue;
        }

        const auto entry = std::find_if(index.plugins.begin(), index.plugins.end(), [&](const auto& item) {
            return item.id == id;
        });

        for (const auto& folder : std::filesystem::directory_iterator(plugin.path(), entryEc)) {
            const auto name = Platform::PathToUtf8(folder.path().filename());
            const bool referenced = entry != index.plugins.end() &&
                                    (entry->folder == name || entry->pending == name);
            if (referenced || folderInUse(folder.path())) {
                continue;
            }

            std::error_code removeEc;
            std::filesystem::remove_all(folder.path(), removeEc);
            if (removeEc) {
                JST_WARN("[PLUGIN] Failed to remove old plugin files '{}'. Retrying on the next start.",
                         Platform::PathToUtf8(folder.path()));
            }
        }

        std::error_code emptyEc;
        if (std::filesystem::is_empty(plugin.path(), emptyEc)) {
            std::filesystem::remove(plugin.path(), emptyEc);
        }
    }
}

Result Plugin::Impl::cacheDirectory(std::filesystem::path& directory) {
    std::lock_guard<std::mutex> guard(cacheMutex);

#if defined(JST_OS_BROWSER)
    if (cacheRunDirectory.empty()) {
        std::error_code ec;
        cacheRunDirectory = std::filesystem::temp_directory_path(ec) / "cyberether-plugins";
        if (ec) {
            JST_ERROR("[PLUGIN] Failed to resolve the temporary plugin cache directory.");
            cacheRunDirectory.clear();
            return Result::ERROR;
        }
    }
#else
    if (cacheRunDirectory.empty()) {
        std::string cachePath;
        JST_CHECK(Platform::CachePath(cachePath));

        const auto cacheRoot = Platform::PathFromUtf8(cachePath) / "plugins";
        const auto runs = cacheRoot / "runs";

        std::error_code ec;
        std::filesystem::create_directories(runs, ec);
        if (ec) {
            JST_ERROR("[PLUGIN] Failed to create plugin cache directory '{}'.", Platform::PathToUtf8(runs));
            return Result::ERROR;
        }

        Platform::FileLock maintenanceLock;
        JST_CHECK(maintenanceLock.acquire(Platform::PathToUtf8(cacheRoot / "maintenance.lock")));
        sweepCache(runs);

        const auto candidate = runs / uniqueStamp();
        std::filesystem::create_directories(candidate, ec);
        if (ec) {
            JST_ERROR("[PLUGIN] Failed to create plugin cache run directory '{}'.", Platform::PathToUtf8(candidate));
            return Result::ERROR;
        }

        if (cacheOwnerLock.acquire(Platform::PathToUtf8(candidate / "owner.lock")) != Result::SUCCESS) {
            std::filesystem::remove_all(candidate, ec);
            return Result::ERROR;
        }

        cacheRunDirectory = candidate;
    }
#endif

    directory = cacheRunDirectory / std::to_string(++cacheGeneration);
    return Result::SUCCESS;
}

void Plugin::Impl::sweepCache(const std::filesystem::path& runs) {
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(runs, ec)) {
        std::error_code entryEc;
        if (!entry.is_directory(entryEc)) {
            continue;
        }

        Platform::FileLock staleOwnerLock;
        if (staleOwnerLock.acquire(Platform::PathToUtf8(entry.path() / "owner.lock"), false) != Result::SUCCESS) {
            continue;
        }
        staleOwnerLock.release();

        std::filesystem::remove_all(entry.path(), entryEc);
        if (entryEc) {
            JST_WARN("[PLUGIN] Failed to remove stale plugin cache directory '{}'.",
                     Platform::PathToUtf8(entry.path()));
        }
    }
}

void Plugin::Impl::cleanupCache() {
    std::lock_guard<std::mutex> guard(cacheMutex);
    if (cacheRunDirectory.empty()) {
        return;
    }

    cacheOwnerLock.release();

    std::error_code ec;
    std::filesystem::remove_all(cacheRunDirectory, ec);
    if (ec) {
        JST_WARN("[PLUGIN] Failed to remove plugin cache run directory '{}'.",
                 Platform::PathToUtf8(cacheRunDirectory));
    }

    cacheRunDirectory.clear();
    cacheGeneration = 0;
}

bool Plugin::Impl::loaded(const std::string& id) {
    std::lock_guard<std::mutex> guard(pluginsMutex);
    return std::any_of(plugins.begin(), plugins.end(), [&](const auto& plugin) {
        return plugin.manifest.id == id;
    });
}

Result Plugin::Impl::open(Record& plugin) {
    JST_CHECK(readManifest(plugin.root, plugin.manifest));
    JST_CHECK(validateManifest(plugin.root, plugin.manifest));

    const auto id = plugin.manifest.id;
    if (loaded(id)) {
        JST_ERROR("[PLUGIN] Plugin '{}' is already loaded.", id);
        return Result::ERROR;
    }

    {
        std::lock_guard<std::mutex> guard(pathsMutex);
        paths[id] = plugin.root;
    }

    const auto before = snapshotRegistry();
    auto fail = [&]() {
        (void)Registry::DiscardStaticRegistrations();
        rollbackRegistrations(diffRegistrations(before, snapshotRegistry()));
        plugin.registrations = {};
        if (plugin.handle != nullptr) {
            Platform::CloseDynamicLibrary(plugin.handle);
            plugin.handle = nullptr;
        }

        std::lock_guard<std::mutex> guard(pathsMutex);
        paths.erase(id);
        return Result::ERROR;
    };

    const auto library = Platform::PathToUtf8(libraryPath(plugin.root, id));
    std::string error;
    try {
        plugin.handle = Platform::OpenDynamicLibrary(library,
                                                     Platform::DynamicLibraryVisibility::Local,
                                                     error,
                                                     Platform::DynamicLibrarySearch::LibraryDirectory);
    } catch (const Result& status) {
        JST_ERROR("[PLUGIN] Exception while loading plugin '{}': {}", id, status);
        return fail();
    } catch (const std::exception& e) {
        JST_ERROR("[PLUGIN] Exception while loading plugin '{}': {}", id, e.what());
        return fail();
    } catch (...) {
        JST_ERROR("[PLUGIN] Unknown exception while loading plugin '{}'.", id);
        return fail();
    }

    if (plugin.handle == nullptr) {
        JST_ERROR("[PLUGIN] Failed to load plugin '{}': {}", id, error);
        return fail();
    }

    if (Registry::DrainStaticRegistrations() != Result::SUCCESS) {
        return fail();
    }

    plugin.registrations = diffRegistrations(before, snapshotRegistry());
    if (validatePrefixes(id, plugin.registrations, before) != Result::SUCCESS ||
        publishExamples(plugin, plugin.examples) != Result::SUCCESS) {
        return fail();
    }

    return Result::SUCCESS;
}

void Plugin::Impl::close(Record& plugin) {
    if (!plugin.examples.empty()) {
        (void)Home::Get().gallery().publish(galleryOrigin(plugin.manifest.id), {});
        plugin.examples.clear();
    }

    rollbackRegistrations(plugin.registrations);
    plugin.registrations = {};

    if (plugin.handle != nullptr) {
        Platform::CloseDynamicLibrary(plugin.handle);
        plugin.handle = nullptr;
    }

    std::lock_guard<std::mutex> guard(pathsMutex);
    const auto current = paths.find(plugin.manifest.id);
    if (current != paths.end() && current->second == plugin.root) {
        paths.erase(current);
    }
}

Result Plugin::Impl::installEntry(const Gallery::Source& source,
                                  const Gallery::Entry& entry,
                                  bool update,
                                  const std::function<bool()>& cancelled) {
    std::filesystem::path directory;
    JST_CHECK(cacheDirectory(directory));
    const auto bundle = directory / "bundle.cep";

    std::string id;
    const auto result = Gallery::Download(source, entry, bundle, cancelled) == Result::SUCCESS
        ? install(Platform::PathToUtf8(bundle),
                  id,
                  {.gallery = true, .update = update, .id = entry.id, .version = entry.version})
        : Result::ERROR;

    std::error_code ec;
    std::filesystem::remove_all(directory, ec);
    return result;
}

Result Plugin::Impl::install(const std::string& path, std::string& id, const InstallOptions& options) {
    id.clear();

    if (!Plugin::IsCepPath(path)) {
        JST_ERROR("[PLUGIN] Plugin '{}' is not a .cep bundle.", path);
        return Result::ERROR;
    }

    std::error_code ec;
    const auto source = Platform::PathFromUtf8(path);
    if (!std::filesystem::is_regular_file(source, ec)) {
        JST_ERROR("[PLUGIN] Plugin '{}' does not exist.", path);
        return Result::ERROR;
    }

    std::lock_guard<std::mutex> lifecycleGuard(lifecycleMutex);

    std::filesystem::path root;
    JST_CHECK(storeRoot(root));

    Platform::FileLock storeLock;
    JST_CHECK(storeLock.acquire(Platform::PathToUtf8(root / "store.lock")));

    Index index;
    JST_CHECK(readIndex(root, index));
    if (!std::filesystem::exists(root / IndexFile, ec)) {
        JST_CHECK(writeIndex(root, index));
    }

    const auto staging = root / ".staging" / uniqueStamp();
    auto discard = [&]() {
        std::error_code cleanupEc;
        std::filesystem::remove_all(staging, cleanupEc);
        return Result::ERROR;
    };

    Manifest manifest;
    if (extract(source, staging) != Result::SUCCESS ||
        readManifest(staging, manifest) != Result::SUCCESS ||
        validateManifest(staging, manifest) != Result::SUCCESS) {
        return discard();
    }

    if ((!options.id.empty() && manifest.id != options.id) ||
        (!options.version.empty() && manifest.version != options.version)) {
        JST_ERROR("[PLUGIN] The downloaded bundle is plugin '{}' version {}, but the gallery listed '{}' version {}.",
                  manifest.id,
                  manifest.version,
                  options.id,
                  options.version);
        return discard();
    }

    const auto folder = manifest.version + "-" + manifest.cyberether + "-" + uniqueStamp();
    const auto destination = root / Platform::PathFromUtf8(manifest.id) / Platform::PathFromUtf8(folder);

    std::filesystem::create_directories(destination.parent_path(), ec);
    if (!ec) {
        std::filesystem::rename(staging, destination, ec);
    }
    if (ec) {
        JST_ERROR("[PLUGIN] Failed to move plugin '{}' into the store.", manifest.id);
        return discard();
    }

    const bool active = loaded(manifest.id);
    auto entry = std::find_if(index.plugins.begin(), index.plugins.end(), [&](const auto& item) {
        return item.id == manifest.id;
    });
    if (options.update && (entry == index.plugins.end() || !entry->gallery)) {
        JST_WARN("[PLUGIN] Plugin '{}' was removed or replaced while its update downloaded. Skipping the update.",
                 manifest.id);
        std::error_code cleanupEc;
        std::filesystem::remove_all(destination, cleanupEc);
        return Result::ERROR;
    }
    if (entry == index.plugins.end() && active) {
        std::string current;
        {
            std::lock_guard<std::mutex> guard(pluginsMutex);
            for (const auto& plugin : plugins) {
                if (!plugin.temporary && plugin.manifest.id == manifest.id) {
                    current = Platform::PathToUtf8(plugin.root.filename());
                }
            }
        }
        if (current.empty()) {
            index.plugins.push_back({.id = manifest.id, .folder = folder, .pending = {}, .gallery = options.gallery});
        } else {
            index.plugins.push_back({
                .id = manifest.id,
                .folder = current,
                .pending = folder,
                .gallery = options.gallery,
            });
        }
    } else if (entry == index.plugins.end()) {
        index.plugins.push_back({.id = manifest.id, .folder = folder, .pending = {}, .gallery = options.gallery});
    } else if (active) {
        entry->pending = folder;
        entry->gallery = options.gallery;
    } else {
        entry->folder = folder;
        entry->pending.clear();
        entry->gallery = options.gallery;
    }

    JST_CHECK(writeIndex(root, index));
    sweepStore(root, index);

    {
        std::lock_guard<std::mutex> guard(pluginsMutex);
        failed.erase(manifest.id);
    }

    id = manifest.id;
    JST_INFO("[PLUGIN] Installed plugin '{}' version {}.", manifest.id, manifest.version);
    return Result::SUCCESS;
}

Result Plugin::Impl::remove(const std::string& id) {
    std::lock_guard<std::mutex> lifecycleGuard(lifecycleMutex);

    std::filesystem::path root;
    JST_CHECK(storeRoot(root));

    Platform::FileLock storeLock;
    JST_CHECK(storeLock.acquire(Platform::PathToUtf8(root / "store.lock")));

    Index index;
    JST_CHECK(readIndex(root, index));

    const auto count = index.plugins.size();
    std::erase_if(index.plugins, [&](const auto& entry) {
        return entry.id == id;
    });
    if (index.plugins.size() == count) {
        JST_ERROR("[PLUGIN] Plugin '{}' is not installed.", id);
        return Result::ERROR;
    }

    JST_CHECK(writeIndex(root, index));
    sweepStore(root, index);

    {
        std::lock_guard<std::mutex> guard(pluginsMutex);
        failed.erase(id);
    }

    JST_INFO("[PLUGIN] Removed plugin '{}'.", id);
    return Result::SUCCESS;
}

Result Plugin::Impl::loadInstalled() {
    std::lock_guard<std::mutex> lifecycleGuard(lifecycleMutex);
    JST_CHECK(Registry::DrainStaticRegistrations());

    std::filesystem::path root;
    JST_CHECK(storeRoot(root));

    Platform::FileLock storeLock;
    JST_CHECK(storeLock.acquire(Platform::PathToUtf8(root / "store.lock")));

    Index index;
    JST_CHECK(readIndex(root, index));

    bool changed = false;
    for (auto& entry : index.plugins) {
        if (!entry.pending.empty() && !loaded(entry.id)) {
            entry.folder = entry.pending;
            entry.pending.clear();
            changed = true;
        }
    }
    if (changed) {
        JST_CHECK(writeIndex(root, index));
    }
    sweepStore(root, index);

    for (const auto& entry : index.plugins) {
        if (loaded(entry.id)) {
            continue;
        }

        Record plugin;
        plugin.root = root / Platform::PathFromUtf8(entry.id) / Platform::PathFromUtf8(entry.folder);

        Manifest manifest;
        if (readManifest(plugin.root, manifest) != Result::SUCCESS || manifest.id != entry.id) {
            JST_WARN("[PLUGIN] Installed plugin '{}' is damaged. Install it again.", entry.id);
            std::lock_guard<std::mutex> guard(pluginsMutex);
            failed.insert(entry.id);
            continue;
        }

        if (!compatible(plugin.root, manifest)) {
            JST_WARN("[PLUGIN] Installed plugin '{}' was built for CyberEther {} ({}). "
                     "Install a build for CyberEther {} ({}).",
                     entry.id,
                     manifest.cyberether,
                     manifest.build,
                     JETSTREAM_VERSION_STR,
                     JETSTREAM_BUILD_ID);
            continue;
        }

        pruneUsers(plugin.root);
        const auto userLock = plugin.root / ".users" / (uniqueStamp() + ".lock");
        if (plugin.user.acquire(Platform::PathToUtf8(userLock)) != Result::SUCCESS ||
            open(plugin) != Result::SUCCESS) {
            JST_WARN("[PLUGIN] Failed to load installed plugin '{}'. Continuing startup.", entry.id);
            plugin.user.release();
            std::error_code ec;
            std::filesystem::remove(userLock, ec);
            std::lock_guard<std::mutex> guard(pluginsMutex);
            failed.insert(entry.id);
            continue;
        }

        std::lock_guard<std::mutex> guard(pluginsMutex);
        failed.erase(entry.id);
        plugins.push_back(std::move(plugin));
    }

    return Result::SUCCESS;
}

Result Plugin::Impl::load(const std::string& path) {
    if (!Plugin::IsCepPath(path)) {
        JST_ERROR("[PLUGIN] Plugin '{}' is not a .cep bundle.", path);
        return Result::ERROR;
    }

    std::error_code ec;
    const auto source = Platform::PathFromUtf8(path);
    if (!std::filesystem::is_regular_file(source, ec)) {
        JST_ERROR("[PLUGIN] Plugin '{}' does not exist.", path);
        return Result::ERROR;
    }

    auto normalized = std::filesystem::weakly_canonical(source, ec);
    if (ec) {
        normalized = std::filesystem::absolute(source, ec);
    }
    const auto sourcePath = Platform::PathToUtf8(normalized);

    std::lock_guard<std::mutex> lifecycleGuard(lifecycleMutex);
    JST_CHECK(Registry::DrainStaticRegistrations());

    {
        std::lock_guard<std::mutex> guard(pluginsMutex);
        const auto duplicate = std::find_if(plugins.begin(), plugins.end(), [&](const auto& entry) {
            return entry.temporary && entry.source == sourcePath;
        });
        if (duplicate != plugins.end()) {
            return Result::SUCCESS;
        }
    }

    Record plugin;
    plugin.source = sourcePath;
    plugin.temporary = true;
    JST_CHECK(cacheDirectory(plugin.root));

    if (extract(normalized, plugin.root) != Result::SUCCESS || open(plugin) != Result::SUCCESS) {
        std::filesystem::remove_all(plugin.root, ec);
        return Result::ERROR;
    }

    std::lock_guard<std::mutex> guard(pluginsMutex);
    plugins.push_back(std::move(plugin));
    return Result::SUCCESS;
}

Result Plugin::Impl::reload(const std::string& id) {
#if defined(JST_OS_BROWSER)
    (void)id;
    JST_ERROR("[PLUGIN] Browser plugins cannot be reloaded in place. Refresh the page instead.");
    return Result::ERROR;
#else
    std::lock_guard<std::mutex> lifecycleGuard(lifecycleMutex);
    JST_CHECK(Registry::DrainStaticRegistrations());

    Record current;
    {
        std::lock_guard<std::mutex> guard(pluginsMutex);
        const auto it = std::find_if(plugins.begin(), plugins.end(), [&](const auto& entry) {
            return entry.manifest.id == id;
        });
        if (it == plugins.end()) {
            JST_ERROR("[PLUGIN] Plugin '{}' is not loaded.", id);
            return Result::ERROR;
        }
        if (!it->temporary) {
            JST_ERROR("[PLUGIN] Plugin '{}' is installed. Only plugins loaded with --plugin can be reloaded.", id);
            return Result::ERROR;
        }
        current = std::move(*it);
        plugins.erase(it);
    }

    close(current);

    Record replacement;
    replacement.source = current.source;
    replacement.temporary = true;

    std::error_code ec;
    const bool reloaded = cacheDirectory(replacement.root) == Result::SUCCESS &&
                          extract(Platform::PathFromUtf8(current.source), replacement.root) == Result::SUCCESS &&
                          open(replacement) == Result::SUCCESS;
    if (!reloaded) {
        if (!replacement.root.empty()) {
            std::filesystem::remove_all(replacement.root, ec);
        }

        Record restored;
        restored.source = current.source;
        restored.temporary = true;
        restored.root = current.root;
        if (open(restored) != Result::SUCCESS) {
            JST_ERROR("[PLUGIN] Failed to restore plugin '{}' after reload failure.", id);
            std::filesystem::remove_all(current.root, ec);
            return Result::ERROR;
        }

        std::lock_guard<std::mutex> guard(pluginsMutex);
        plugins.push_back(std::move(restored));
        return Result::ERROR;
    }

    std::filesystem::remove_all(current.root, ec);

    std::lock_guard<std::mutex> guard(pluginsMutex);
    plugins.push_back(std::move(replacement));
    return Result::SUCCESS;
#endif
}

std::vector<Plugin::Info> Plugin::Impl::list() {
    std::vector<Plugin::Info> rows;
    (void)collect(rows);
    return rows;
}

Result Plugin::Impl::collect(std::vector<Plugin::Info>& rows) {
    rows.clear();

    auto describe = [](const Manifest& manifest, Plugin::Info& info) {
        info.id = manifest.id;
        info.title = manifest.title;
        info.summary = manifest.summary;
        info.description = manifest.description;
        info.author = manifest.author;
        info.license = manifest.license;
        info.version = manifest.version;
        info.cyberether = manifest.cyberether;
    };

    {
        std::lock_guard<std::mutex> guard(pluginsMutex);
        for (const auto& plugin : plugins) {
            if (!plugin.temporary) {
                continue;
            }
            Plugin::Info info;
            describe(plugin.manifest, info);
            info.path = Platform::PathToUtf8(plugin.root);
            info.source = plugin.source;
            info.temporary = true;
            info.status = Status::Loaded;
            info.blocks = plugin.registrations.blocks;
            info.examples = plugin.examples;
            rows.push_back(std::move(info));
        }
    }

    std::filesystem::path root;
    Index index;
    if (storeRoot(root, false) != Result::SUCCESS || readIndex(root, index) != Result::SUCCESS) {
        return Result::ERROR;
    }

    for (const auto& entry : index.plugins) {
        if (std::any_of(rows.begin(), rows.end(), [&](const auto& row) { return row.id == entry.id; })) {
            continue;
        }

        const auto folder = root / Platform::PathFromUtf8(entry.id) / Platform::PathFromUtf8(entry.folder);

        Plugin::Info info;
        Manifest manifest;
        const bool readable = readManifest(folder, manifest) == Result::SUCCESS;
        if (readable) {
            describe(manifest, info);
        }
        info.id = entry.id;
        info.path = Platform::PathToUtf8(folder);
        info.gallery = entry.gallery;

        if (!entry.pending.empty()) {
            const auto pendingFolder = root / Platform::PathFromUtf8(entry.id) / Platform::PathFromUtf8(entry.pending);
            Manifest pending;
            if (readManifest(pendingFolder, pending) == Result::SUCCESS && compatible(pendingFolder, pending)) {
                info.pending = pending.version;
            }
        }

        std::lock_guard<std::mutex> guard(pluginsMutex);
        const auto record = std::find_if(plugins.begin(), plugins.end(), [&](const auto& plugin) {
            return !plugin.temporary && plugin.manifest.id == entry.id;
        });
        if (record != plugins.end()) {
            describe(record->manifest, info);
            info.path = Platform::PathToUtf8(record->root);
            info.status = Status::Loaded;
            info.blocks = record->registrations.blocks;
            info.examples = record->examples;
        } else if (!readable || failed.contains(entry.id)) {
            info.status = Status::Failed;
        } else if (!compatible(folder, manifest)) {
            info.status = entry.gallery ? Status::Waiting : Status::Incompatible;
        } else {
            info.status = Status::Restart;
        }

        rows.push_back(std::move(info));
    }

    return Result::SUCCESS;
}

std::filesystem::path Plugin::Impl::path(const std::string& id) {
    std::lock_guard<std::mutex> guard(pathsMutex);
    const auto entry = paths.find(id);
    return entry == paths.end() ? std::filesystem::path() : entry->second;
}

std::string Plugin::Impl::owner(const std::string& block, const std::string& provider) {
    std::lock_guard<std::mutex> guard(pluginsMutex);
    for (const auto& plugin : plugins) {
        const auto& registrations = plugin.registrations;
        if (std::find(registrations.blocks.begin(), registrations.blocks.end(), block) != registrations.blocks.end()) {
            return plugin.manifest.id;
        }

        if (!provider.starts_with(plugin.manifest.id + "_")) {
            continue;
        }

        const auto& modules = registrations.modules;
        const auto module = std::find_if(modules.begin(), modules.end(), [&](const auto& entry) {
            return entry.provider == provider;
        });
        if (module != modules.end()) {
            return plugin.manifest.id;
        }
    }

    return {};
}

void Plugin::Impl::cleanup() {
    std::lock_guard<std::mutex> lifecycleGuard(lifecycleMutex);

    std::vector<Record> records;
    {
        std::lock_guard<std::mutex> guard(pluginsMutex);
        records.swap(plugins);
    }

    for (auto it = records.rbegin(); it != records.rend(); ++it) {
        close(*it);
        if (it->temporary) {
            std::error_code ec;
            std::filesystem::remove_all(it->root, ec);
        }
    }

    cleanupCache();
}

Plugin::Impl& Plugin::plugin() {
    static Impl impl;
    return impl;
}

bool Plugin::IsCepPath(const std::string& path) {
    auto extension = Platform::PathToUtf8(Platform::PathFromUtf8(path).extension());
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return extension == ".cep";
}

Result Plugin::Install(const std::string& path, std::string& id) {
    return plugin().install(path, id);
}

Result Plugin::Gallery::Install(const Source& source,
                               const std::string& id,
                               const std::function<bool()>& cancelled) {
    if (!ValidId(id)) {
        JST_ERROR("[PLUGIN] '{}' is not a plugin ID.", id);
        return Result::ERROR;
    }

    Entries entries;
    JST_CHECK(Fetch(source, Current({id}), cancelled, entries));

    const auto entry = std::find_if(entries.begin(), entries.end(), [&](const auto& item) {
        return item.id == id;
    });
    if (entry == entries.end()) {
        JST_ERROR("[PLUGIN] Plugin '{}' is not in the CyberEther gallery.", id);
        return Result::ERROR;
    }
    if (!entry->available()) {
        JST_ERROR("[PLUGIN] Plugin '{}' does not have a build for CyberEther {} ({}) yet.",
                  id,
                  JETSTREAM_VERSION_STR,
                  JETSTREAM_BUILD_ID);
        return Result::ERROR;
    }

    return plugin().installEntry(source, *entry, false, cancelled);
}

bool Plugin::Gallery::Outdated(const Info& info, const Entry& entry) {
    if (info.temporary || !info.gallery || info.id != entry.id || !entry.available() ||
        info.pending == entry.version) {
        return false;
    }
    return info.version != entry.version || info.status == Status::Waiting;
}

Result Plugin::Gallery::Updates(const Source& source,
                               Entries& updates,
                               const std::function<bool()>& cancelled) {
    updates.clear();

    std::vector<Info> inventory;
    JST_CHECK(plugin().collect(inventory));

    std::vector<Info> installed;
    std::vector<std::string> ids;
    for (auto& info : inventory) {
        if (!info.temporary && info.gallery) {
            ids.push_back(info.id);
            installed.push_back(std::move(info));
        }
    }
    if (ids.empty()) {
        return Result::SUCCESS;
    }

    Entries entries;
    JST_CHECK(Fetch(source, Current(ids), cancelled, entries));

    for (const auto& info : installed) {
        const auto entry = std::find_if(entries.begin(), entries.end(), [&](const auto& item) {
            return item.id == info.id;
        });
        if (entry == entries.end()) {
            JST_WARN("[PLUGIN] Plugin '{}' is no longer in the CyberEther gallery.", info.id);
            continue;
        }
        if (Outdated(info, *entry)) {
            updates.push_back(*entry);
        }
    }

    return Result::SUCCESS;
}

Result Plugin::Gallery::Update(const Source& source,
                              const std::vector<std::string>& ids,
                              std::vector<std::string>& updated,
                              const std::function<bool()>& cancelled) {
    updated.clear();

    Entries updates;
    JST_CHECK(Updates(source, updates, cancelled));

    Result result = Result::SUCCESS;
    for (const auto& entry : updates) {
        if (cancelled && cancelled()) {
            break;
        }
        if (std::find(ids.begin(), ids.end(), entry.id) == ids.end()) {
            continue;
        }

        if (plugin().installEntry(source, entry, true, cancelled) == Result::SUCCESS) {
            updated.push_back(entry.id);
        } else {
            JST_ERROR("[PLUGIN] Failed to update plugin '{}' from the gallery.", entry.id);
            result = Result::ERROR;
        }
    }

    return result;
}

Result Plugin::Remove(const std::string& id) {
    return plugin().remove(id);
}

Result Plugin::LoadInstalled() {
    return plugin().loadInstalled();
}

Result Plugin::Load(const std::string& path) {
    return plugin().load(path);
}

Result Plugin::Reload(const std::string& id) {
    return plugin().reload(id);
}

std::vector<Plugin::Info> Plugin::List() {
    return plugin().list();
}

std::filesystem::path Plugin::Path(const std::string& id) {
    return plugin().path(id);
}

std::string Plugin::Owner(const std::string& block, const std::string& provider) {
    return plugin().owner(block, provider);
}

}  // namespace Jetstream
