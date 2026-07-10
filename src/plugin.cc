#include "jetstream/plugin.hh"

#include "jetstream/logger.hh"
#include "jetstream/platform.hh"
#include "jetstream/parser.hh"
#include "jetstream/registry.hh"
#include "jetstream/runtime.hh"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <zlib.h>

namespace Jetstream {

struct Plugin::Impl {
    struct ManifestMetadata {
        std::string name;
        std::string version;
        std::string minimumJetstreamVersion;

        JST_SERDES(name, version, minimumJetstreamVersion);
    };

    struct ManifestTarget {
        std::string path;
        std::string system;
        std::string device;
        std::string arch;

        JST_SERDES(path, system, device, arch);
    };

    struct ManifestExample {
        std::string path;

        JST_SERDES(path);
    };

    struct Manifest {
        ManifestMetadata metadata;
        std::vector<ManifestTarget> targets;
        std::vector<ManifestExample> examples;

        JST_SERDES(metadata, targets, examples);
    };

    struct Registrations {
        struct Module {
            std::string type;
            DeviceType device = DeviceType::None;
            RuntimeType runtime = RuntimeType::NONE;
            ProviderType provider;
        };

        std::vector<Module> modules;
        std::vector<std::string> blocks;
        std::vector<std::string> flowgraphs;

        struct Benchmark {
            std::string moduleType;
            const void* owner = nullptr;
        };

        std::vector<Benchmark> benchmarks;
    };

    struct Record {
        std::string sourcePath;
        std::string loadedPath;
        std::string extractedPath;
        std::vector<void*> handles;
        Registrations registrations;
        Manifest manifest;
        std::vector<std::string> loadedTargets;
    };

    struct RegistrySnapshot {
        std::vector<Registrations::Module> modules;
        std::vector<std::string> blocks;
        std::vector<std::string> flowgraphs;
        std::vector<Registrations::Benchmark> benchmarks;
    };

    ~Impl();

    Result load(const std::string& path);
    Result reload(const std::string& path);
    std::vector<Plugin::Info> list();

 private:
    static std::string normalizePath(const std::string& path);
    static std::string cacheFileName(const std::string& sourcePath, uint64_t generation);
    static std::string lowercase(std::string value);
    static std::string currentSystem();
    static std::string currentArch();
    static bool isCepPath(const std::string& path);
    static bool isDeviceAvailable(DeviceType device);
    static bool parseVersion(const std::string& version, uint32_t& encoded);
    static bool safeRelativePath(const std::string& rawPath, std::filesystem::path& relativePath);

    static void closeHandle(void* handle);
    static const JetstreamPluginAbi* loadAbi(void* handle, std::string& error);

    static Result readFileBytes(const std::filesystem::path& path, std::vector<uint8_t>& bytes);
    static Result readTextFile(const std::filesystem::path& path, std::string& content);
    static Result decompressGzip(const std::vector<uint8_t>& compressed, std::vector<uint8_t>& decompressed);
    static Result extractCepArchive(const std::string& sourcePath, const std::filesystem::path& destination);
    static Result loadManifest(const std::filesystem::path& bundlePath, Manifest& manifest);
    static Result validateManifest(const std::string& sourcePath, const Manifest& manifest);
    static bool isCompatibleTarget(const ManifestTarget& target);
    static Plugin::Info buildInfo(const Record& plugin);

    static bool containsModule(const std::vector<Registrations::Module>& modules,
                               const Registrations::Module& module);
    static bool containsBenchmark(const std::vector<Registrations::Benchmark>& benchmarks,
                                  const Registrations::Benchmark& benchmark);
    static bool containsString(const std::vector<std::string>& values,
                               const std::string& value);
    static RegistrySnapshot snapshotRegistry();
    static Registrations diffRegistrations(const RegistrySnapshot& before,
                                           const RegistrySnapshot& after);
    static void rollbackRegistrations(const Registrations& registrations);

    Result copyToCache(const std::string& sourcePath, std::string& cachedPath);
    Result ensureCacheReady();
    Result createCacheRunDirectory(const std::filesystem::path& runsDirectory);
    Result extractToCache(const std::string& sourcePath, std::string& extractedPath);
    void sweepCache(const std::filesystem::path& runsDirectory);
    void cleanupCache();

    Result loadPluginCopy(const std::string& sourcePath, Record& plugin);
    Result openPlugin(const std::string& path, void*& handle);
    Result validatePluginAbi(const std::string& path, void* handle);
    Result registerExamples(const std::filesystem::path& bundlePath, const Manifest& manifest);
    void closePlugin(Record& plugin, bool removeCachedFile = true);
    void removeCachedPluginFiles(Record& plugin);
    void cleanup();

    std::mutex cacheMutex;
    std::mutex lifecycleMutex;
    std::mutex pluginsMutex;

    Platform::FileLock cacheOwnerLock;
    std::filesystem::path cacheRunDirectory;
    uint64_t cacheGeneration = 0;
    std::vector<Record> plugins;
};

Plugin::Impl::~Impl() {
    cleanup();
}

std::string Plugin::Impl::normalizePath(const std::string& path) {
    std::error_code ec;
    const auto inputPath = Platform::PathFromUtf8(path);

    const auto canonicalPath = std::filesystem::weakly_canonical(inputPath, ec);
    if (!ec) {
        return Platform::PathToUtf8(canonicalPath);
    }

    ec.clear();
    const auto absolutePath = std::filesystem::absolute(inputPath, ec);
    if (!ec) {
        return Platform::PathToUtf8(absolutePath);
    }

    return path;
}

std::string Plugin::Impl::cacheFileName(const std::string& sourcePath, uint64_t generation) {
    auto filename = Platform::PathToUtf8(Platform::PathFromUtf8(sourcePath).filename());
    if (filename.empty()) {
        filename = "plugin";
    }

    const auto sourceHash = std::hash<std::string>{}(sourcePath);
    return std::to_string(sourceHash) + "-" + std::to_string(generation) + "-" + filename;
}

std::string Plugin::Impl::lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

std::string Plugin::Impl::currentSystem() {
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

std::string Plugin::Impl::currentArch() {
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

bool Plugin::Impl::isCepPath(const std::string& path) {
    return lowercase(std::filesystem::path(path).extension().string()) == ".cep";
}

bool Plugin::Impl::isDeviceAvailable(DeviceType device) {
    switch (device) {
        case DeviceType::CPU:
#if defined(JETSTREAM_BACKEND_CPU_AVAILABLE)
            return true;
#else
            return false;
#endif
        case DeviceType::CUDA:
#if defined(JETSTREAM_BACKEND_CUDA_AVAILABLE)
            return true;
#else
            return false;
#endif
        case DeviceType::Metal:
#if defined(JETSTREAM_BACKEND_METAL_AVAILABLE)
            return true;
#else
            return false;
#endif
        case DeviceType::Vulkan:
#if defined(JETSTREAM_BACKEND_VULKAN_AVAILABLE)
            return true;
#else
            return false;
#endif
        case DeviceType::WebGPU:
#if defined(JETSTREAM_BACKEND_WEBGPU_AVAILABLE)
            return true;
#else
            return false;
#endif
        default:
            return false;
    }
}

bool Plugin::Impl::parseVersion(const std::string& version, uint32_t& encoded) {
    const auto parts = Parser::SplitString(version, ".");
    if (parts.size() != 3) {
        return false;
    }

    uint32_t values[3] = {0, 0, 0};
    for (std::size_t i = 0; i < 3; ++i) {
        if (parts[i].empty()) {
            return false;
        }

        std::size_t consumed = 0;
        unsigned long value = 0;
        try {
            value = std::stoul(parts[i], &consumed, 10);
        } catch (...) {
            return false;
        }

        if (consumed != parts[i].size() || value > 255) {
            return false;
        }

        values[i] = static_cast<uint32_t>(value);
    }

    encoded = JETSTREAM_VERSION_ENCODE(values[0], values[1], values[2]);
    return true;
}

bool Plugin::Impl::safeRelativePath(const std::string& rawPath, std::filesystem::path& relativePath) {
    if (rawPath.empty() ||
        rawPath.find('\\') != std::string::npos ||
        rawPath.find(':') != std::string::npos) {
        return false;
    }

    std::filesystem::path candidate(rawPath);
    if (candidate.is_absolute() || candidate.has_root_name()) {
        return false;
    }

    candidate = candidate.lexically_normal();
    if (candidate.empty() || candidate == ".") {
        return false;
    }

    for (const auto& part : candidate) {
        if (part == "..") {
            return false;
        }
    }

    relativePath = std::move(candidate);
    return true;
}

void Plugin::Impl::closeHandle(void* handle) {
    Platform::CloseDynamicLibrary(handle);
}

const JetstreamPluginAbi* Plugin::Impl::loadAbi(void* handle, std::string& error) {
    return reinterpret_cast<const JetstreamPluginAbi*>(
        Platform::LoadDynamicLibrarySymbol(handle, JETSTREAM_PLUGIN_ABI_SYMBOL, error));
}

Result Plugin::Impl::readFileBytes(const std::filesystem::path& path, std::vector<uint8_t>& bytes) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        JST_ERROR("[PLUGIN] Failed to open '{}'.", path.string());
        return Result::ERROR;
    }

    file.seekg(0, std::ios::end);
    const auto size = file.tellg();
    if (size < 0) {
        JST_ERROR("[PLUGIN] Failed to determine size for '{}'.", path.string());
        return Result::ERROR;
    }

    file.seekg(0, std::ios::beg);
    bytes.resize(static_cast<std::size_t>(size));
    if (!bytes.empty()) {
        file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        if (!file) {
            JST_ERROR("[PLUGIN] Failed to read '{}'.", path.string());
            return Result::ERROR;
        }
    }

    return Result::SUCCESS;
}

Result Plugin::Impl::readTextFile(const std::filesystem::path& path, std::string& content) {
    std::vector<uint8_t> bytes;
    JST_CHECK(readFileBytes(path, bytes));
    if (bytes.empty()) {
        content.clear();
        return Result::SUCCESS;
    }

    content.assign(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    return Result::SUCCESS;
}

Result Plugin::Impl::decompressGzip(const std::vector<uint8_t>& compressed,
                                    std::vector<uint8_t>& decompressed) {
    if (compressed.empty() || compressed.size() > std::numeric_limits<uInt>::max()) {
        JST_ERROR("[PLUGIN] Invalid gzip payload size.");
        return Result::ERROR;
    }

    z_stream stream = {};
    stream.next_in = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(compressed.data()));
    stream.avail_in = static_cast<uInt>(compressed.size());

    if (inflateInit2(&stream, 15 + 16) != Z_OK) {
        JST_ERROR("[PLUGIN] Failed to initialize gzip decompressor.");
        return Result::ERROR;
    }

    std::array<uint8_t, 64 * 1024> buffer;
    int status = Z_OK;
    while (status == Z_OK) {
        stream.next_out = reinterpret_cast<Bytef*>(buffer.data());
        stream.avail_out = static_cast<uInt>(buffer.size());

        status = inflate(&stream, Z_NO_FLUSH);
        const auto produced = buffer.size() - stream.avail_out;
        decompressed.insert(decompressed.end(), buffer.begin(), buffer.begin() + produced);
    }

    inflateEnd(&stream);

    if (status != Z_STREAM_END) {
        JST_ERROR("[PLUGIN] Failed to decompress gzip payload.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result Plugin::Impl::extractCepArchive(const std::string& sourcePath,
                                       const std::filesystem::path& destination) {
    std::vector<uint8_t> compressed;
    JST_CHECK(readFileBytes(std::filesystem::path(sourcePath), compressed));

    std::vector<uint8_t> tar;
    JST_CHECK(decompressGzip(compressed, tar));

    std::error_code ec;
    std::filesystem::create_directories(destination, ec);
    if (ec) {
        JST_ERROR("[PLUGIN] Failed to create plugin extraction directory '{}'.", destination.string());
        return Result::ERROR;
    }

    auto parseOctal = [](const uint8_t* data, std::size_t size, uint64_t& value) {
        value = 0;
        bool sawDigit = false;

        for (std::size_t i = 0; i < size; ++i) {
            const auto ch = data[i];
            if (ch == 0 || ch == ' ') {
                continue;
            }
            if (ch < '0' || ch > '7') {
                return false;
            }

            sawDigit = true;
            value = (value << 3) + static_cast<uint64_t>(ch - '0');
        }

        return sawDigit;
    };

    auto tarString = [](const uint8_t* data, std::size_t size) {
        std::size_t end = 0;
        while (end < size && data[end] != 0) {
            ++end;
        }

        while (end > 0 && data[end - 1] == ' ') {
            --end;
        }

        return std::string(reinterpret_cast<const char*>(data), end);
    };

    std::size_t offset = 0;
    while (offset + 512 <= tar.size()) {
        const auto* header = tar.data() + offset;
        const bool emptyBlock = std::all_of(header, header + 512, [](uint8_t value) {
            return value == 0;
        });
        if (emptyBlock) {
            return Result::SUCCESS;
        }

        uint64_t storedChecksum = 0;
        if (!parseOctal(header + 148, 8, storedChecksum)) {
            JST_ERROR("[PLUGIN] Invalid tar checksum in '{}'.", sourcePath);
            return Result::ERROR;
        }

        uint64_t computedChecksum = 0;
        for (std::size_t i = 0; i < 512; ++i) {
            computedChecksum += (i >= 148 && i < 156) ? ' ' : header[i];
        }

        if (storedChecksum != computedChecksum) {
            JST_ERROR("[PLUGIN] Tar checksum mismatch in '{}'.", sourcePath);
            return Result::ERROR;
        }

        std::string name = tarString(header, 100);
        const std::string prefix = tarString(header + 345, 155);
        if (!prefix.empty()) {
            name = prefix + "/" + name;
        }

        uint64_t entrySize = 0;
        if (!parseOctal(header + 124, 12, entrySize)) {
            JST_ERROR("[PLUGIN] Invalid tar entry size for '{}'.", name);
            return Result::ERROR;
        }

        const auto type = header[156] == 0 ? '0' : static_cast<char>(header[156]);
        const auto dataOffset = offset + 512;
        const auto paddedSize = ((entrySize + 511) / 512) * 512;
        if (entrySize > std::numeric_limits<std::size_t>::max() ||
            dataOffset + entrySize > tar.size() ||
            dataOffset + paddedSize > tar.size()) {
            JST_ERROR("[PLUGIN] Tar entry '{}' extends beyond archive bounds.", name);
            return Result::ERROR;
        }

        std::filesystem::path relativePath;
        if (!safeRelativePath(name, relativePath)) {
            JST_ERROR("[PLUGIN] Unsafe tar entry path '{}'.", name);
            return Result::ERROR;
        }

        const auto outputPath = destination / relativePath;
        if (type == '5') {
            std::filesystem::create_directories(outputPath, ec);
            if (ec) {
                JST_ERROR("[PLUGIN] Failed to create directory '{}'.", outputPath.string());
                return Result::ERROR;
            }
        } else if (type == '0') {
            std::filesystem::create_directories(outputPath.parent_path(), ec);
            if (ec) {
                JST_ERROR("[PLUGIN] Failed to create directory '{}'.", outputPath.parent_path().string());
                return Result::ERROR;
            }

            std::ofstream file(outputPath, std::ios::binary | std::ios::trunc);
            if (!file) {
                JST_ERROR("[PLUGIN] Failed to create file '{}'.", outputPath.string());
                return Result::ERROR;
            }

            if (entrySize > 0) {
                file.write(reinterpret_cast<const char*>(tar.data() + dataOffset),
                           static_cast<std::streamsize>(entrySize));
                if (!file) {
                    JST_ERROR("[PLUGIN] Failed to write file '{}'.", outputPath.string());
                    return Result::ERROR;
                }
            }
        } else {
            JST_ERROR("[PLUGIN] Unsupported tar entry type '{}' for '{}'.", type, name);
            return Result::ERROR;
        }

        offset = dataOffset + static_cast<std::size_t>(paddedSize);
    }

    JST_ERROR("[PLUGIN] Tar archive '{}' is missing an end marker.", sourcePath);
    return Result::ERROR;
}

Result Plugin::Impl::loadManifest(const std::filesystem::path& bundlePath, Manifest& manifest) {
    std::string content;
    JST_CHECK(readTextFile(bundlePath / "manifest.yml", content));

    Parser::Map data;
    JST_CHECK(Parser::YamlDecode(content, data));
    JST_CHECK(manifest.deserialize(data));
    return Result::SUCCESS;
}

Result Plugin::Impl::validateManifest(const std::string& sourcePath, const Manifest& manifest) {
    if (manifest.metadata.name.empty() || manifest.metadata.version.empty()) {
        JST_ERROR("[PLUGIN] Plugin '{}' has incomplete metadata.", sourcePath);
        return Result::ERROR;
    }

    uint32_t minimumVersion = 0;
    if (!parseVersion(manifest.metadata.minimumJetstreamVersion, minimumVersion)) {
        JST_ERROR("[PLUGIN] Plugin '{}' has invalid minimumJetstreamVersion '{}'.",
                  sourcePath,
                  manifest.metadata.minimumJetstreamVersion);
        return Result::ERROR;
    }

    if (minimumVersion > JETSTREAM_VERSION_CURRENT) {
        JST_ERROR("[PLUGIN] Plugin '{}' requires Jetstream version {}, current version is {}.",
                  sourcePath,
                  manifest.metadata.minimumJetstreamVersion,
                  JETSTREAM_VERSION_STR);
        return Result::ERROR;
    }

    if (manifest.targets.empty()) {
        JST_ERROR("[PLUGIN] Plugin '{}' does not define any targets.", sourcePath);
        return Result::ERROR;
    }

    for (const auto& target : manifest.targets) {
        std::filesystem::path targetPath;
        if (!safeRelativePath(target.path, targetPath) ||
            target.system.empty() ||
            target.device.empty() ||
            target.arch.empty()) {
            JST_ERROR("[PLUGIN] Plugin '{}' has an invalid target entry.", sourcePath);
            return Result::ERROR;
        }

        if (StringToDevice(lowercase(target.device)) == DeviceType::None) {
            JST_ERROR("[PLUGIN] Plugin '{}' target '{}' has invalid device '{}'.",
                      sourcePath,
                      target.path,
                      target.device);
            return Result::ERROR;
        }
    }

    for (const auto& example : manifest.examples) {
        std::filesystem::path examplePath;
        if (!safeRelativePath(example.path, examplePath)) {
            JST_ERROR("[PLUGIN] Plugin '{}' has an invalid example path '{}'.",
                      sourcePath,
                      example.path);
            return Result::ERROR;
        }
    }

    return Result::SUCCESS;
}

bool Plugin::Impl::isCompatibleTarget(const ManifestTarget& target) {
    const auto device = StringToDevice(lowercase(target.device));
    return lowercase(target.system) == currentSystem() &&
           lowercase(target.arch) == currentArch() &&
           isDeviceAvailable(device);
}

Plugin::Info Plugin::Impl::buildInfo(const Record& plugin) {
    Plugin::Info info;
    info.path = plugin.sourcePath;
    info.name = plugin.manifest.metadata.name;
    info.version = plugin.manifest.metadata.version;
    info.minimumJetstreamVersion = plugin.manifest.metadata.minimumJetstreamVersion;
    info.status = "Loaded";
    info.registeredModules = static_cast<U64>(plugin.registrations.modules.size());
    info.registeredBlocks = static_cast<U64>(plugin.registrations.blocks.size());
    info.registeredExamples = static_cast<U64>(plugin.registrations.flowgraphs.size());
    info.registeredBenchmarks = static_cast<U64>(plugin.registrations.benchmarks.size());

    info.targets.reserve(plugin.manifest.targets.size());
    for (const auto& target : plugin.manifest.targets) {
        info.targets.push_back({
            .path = target.path,
            .system = target.system,
            .device = target.device,
            .arch = target.arch,
            .compatible = isCompatibleTarget(target),
            .loaded = containsString(plugin.loadedTargets, target.path),
        });
    }

    return info;
}

bool Plugin::Impl::containsModule(const std::vector<Registrations::Module>& modules,
                                  const Registrations::Module& module) {
    return std::find_if(modules.begin(), modules.end(), [&](const auto& entry) {
        return entry.type == module.type &&
               entry.device == module.device &&
               entry.runtime == module.runtime &&
               entry.provider == module.provider;
    }) != modules.end();
}

bool Plugin::Impl::containsBenchmark(const std::vector<Registrations::Benchmark>& benchmarks,
                                     const Registrations::Benchmark& benchmark) {
    return std::find_if(benchmarks.begin(), benchmarks.end(), [&](const auto& entry) {
        return entry.moduleType == benchmark.moduleType && entry.owner == benchmark.owner;
    }) != benchmarks.end();
}

bool Plugin::Impl::containsString(const std::vector<std::string>& values,
                                  const std::string& value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}

Plugin::Impl::RegistrySnapshot Plugin::Impl::snapshotRegistry() {
    RegistrySnapshot snapshot;

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

    for (const auto& flowgraph : Registry::ListAvailableFlowgraphs()) {
        snapshot.flowgraphs.push_back(flowgraph.key);
    }

    for (const auto& benchmark : Registry::ListAvailableBenchmarks()) {
        snapshot.benchmarks.push_back({
            .moduleType = benchmark.moduleType,
            .owner = benchmark.owner,
        });
    }

    return snapshot;
}

Plugin::Impl::Registrations Plugin::Impl::diffRegistrations(const RegistrySnapshot& before,
                                                            const RegistrySnapshot& after) {
    Registrations diff;

    for (const auto& module : after.modules) {
        if (!containsModule(before.modules, module)) {
            diff.modules.push_back(module);
        }
    }

    for (const auto& block : after.blocks) {
        if (!containsString(before.blocks, block)) {
            diff.blocks.push_back(block);
        }
    }

    for (const auto& flowgraph : after.flowgraphs) {
        if (!containsString(before.flowgraphs, flowgraph)) {
            diff.flowgraphs.push_back(flowgraph);
        }
    }

    for (const auto& benchmark : after.benchmarks) {
        if (!containsBenchmark(before.benchmarks, benchmark)) {
            diff.benchmarks.push_back(benchmark);
        }
    }

    return diff;
}

void Plugin::Impl::rollbackRegistrations(const Registrations& registrations) {
    for (auto it = registrations.benchmarks.rbegin(); it != registrations.benchmarks.rend(); ++it) {
        (void)Registry::UnregisterBenchmark(it->moduleType, it->owner);
    }

    for (auto it = registrations.flowgraphs.rbegin(); it != registrations.flowgraphs.rend(); ++it) {
        (void)Registry::UnregisterFlowgraph(*it);
    }

    for (auto it = registrations.blocks.rbegin(); it != registrations.blocks.rend(); ++it) {
        (void)Registry::UnregisterBlock(*it);
    }

    for (auto it = registrations.modules.rbegin(); it != registrations.modules.rend(); ++it) {
        (void)Registry::UnregisterModule(it->type, it->device, it->runtime, it->provider);
    }
}

Result Plugin::Impl::copyToCache(const std::string& sourcePath, std::string& cachedPath) {
    std::lock_guard<std::mutex> guard(cacheMutex);
    JST_CHECK(ensureCacheReady());

    const auto destination = cacheRunDirectory / cacheFileName(sourcePath, ++cacheGeneration);

    std::error_code ec;
    std::filesystem::copy_file(Platform::PathFromUtf8(sourcePath),
                               destination,
                               std::filesystem::copy_options::overwrite_existing,
                               ec);
    if (ec) {
        std::error_code cleanupEc;
        (void)std::filesystem::remove(destination, cleanupEc);

        JST_ERROR("[PLUGIN] Failed to copy plugin '{}' to cache '{}'.",
                  sourcePath,
                  destination.string());
        return Result::ERROR;
    }

    cachedPath = Platform::PathToUtf8(destination);
    return Result::SUCCESS;
}

Result Plugin::Impl::ensureCacheReady() {
    if (!cacheRunDirectory.empty()) {
        return Result::SUCCESS;
    }

    std::string cachePath;
    JST_CHECK(Platform::CachePath(cachePath));

    const auto cacheRoot = Platform::PathFromUtf8(cachePath) / "registry-plugins";
    const auto runsDirectory = cacheRoot / "runs";

    std::error_code ec;
    std::filesystem::create_directories(runsDirectory, ec);
    if (ec) {
        JST_ERROR("[PLUGIN] Failed to create plugin cache directory '{}'.", runsDirectory.string());
        return Result::ERROR;
    }

    Platform::FileLock maintenanceLock;
    JST_CHECK(maintenanceLock.acquire(Platform::PathToUtf8(cacheRoot / "maintenance.lock")));

    sweepCache(runsDirectory);
    JST_CHECK(createCacheRunDirectory(runsDirectory));
    return Result::SUCCESS;
}

Result Plugin::Impl::createCacheRunDirectory(const std::filesystem::path& runsDirectory) {
    const auto timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    std::error_code ec;
    for (uint64_t attempt = 0; attempt < 1024; ++attempt) {
        const auto candidate = runsDirectory / (std::to_string(timestamp) + "-" + std::to_string(attempt));

        ec.clear();
        if (std::filesystem::create_directory(candidate, ec)) {
            if (cacheOwnerLock.acquire(Platform::PathToUtf8(candidate / "owner.lock")) != Result::SUCCESS) {
                std::error_code cleanupEc;
                (void)std::filesystem::remove_all(candidate, cleanupEc);
                return Result::ERROR;
            }

            cacheRunDirectory = candidate;
            return Result::SUCCESS;
        }

        if (ec) {
            JST_ERROR("[PLUGIN] Failed to create plugin cache run directory '{}'.", candidate.string());
            return Result::ERROR;
        }
    }

    JST_ERROR("[PLUGIN] Failed to allocate a unique plugin cache run directory.");
    return Result::ERROR;
}

Result Plugin::Impl::extractToCache(const std::string& sourcePath, std::string& extractedPath) {
    std::lock_guard<std::mutex> guard(cacheMutex);
    JST_CHECK(ensureCacheReady());

    auto directoryName = cacheFileName(sourcePath, ++cacheGeneration);
    if (!directoryName.ends_with(".cep")) {
        directoryName += ".cep";
    }
    directoryName += ".contents";

    const auto destination = cacheRunDirectory / directoryName;
    if (extractCepArchive(sourcePath, destination) != Result::SUCCESS) {
        std::error_code cleanupEc;
        (void)std::filesystem::remove_all(destination, cleanupEc);
        return Result::ERROR;
    }

    extractedPath = Platform::PathToUtf8(destination);
    return Result::SUCCESS;
}

void Plugin::Impl::sweepCache(const std::filesystem::path& runsDirectory) {
    std::error_code ec;
    std::filesystem::directory_iterator entries(runsDirectory, ec);
    if (ec) {
        JST_WARN("[PLUGIN] Failed to inspect plugin cache directory '{}'.", runsDirectory.string());
        return;
    }

    for (const auto& entry : entries) {
        ec.clear();
        if (!entry.is_directory(ec) || ec) {
            continue;
        }

        Platform::FileLock staleOwnerLock;
        const auto lockResult = staleOwnerLock.acquire(Platform::PathToUtf8(entry.path() / "owner.lock"), false);
        if (lockResult != Result::SUCCESS) {
            continue;
        }

        staleOwnerLock.release();

        std::error_code cleanupEc;
        (void)std::filesystem::remove_all(entry.path(), cleanupEc);
        if (cleanupEc) {
            JST_WARN("[PLUGIN] Failed to remove stale plugin cache directory '{}'.", entry.path().string());
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
    (void)std::filesystem::remove_all(cacheRunDirectory, ec);
    if (ec) {
        JST_WARN("[PLUGIN] Failed to remove plugin cache run directory '{}'.",
                 cacheRunDirectory.string());
    }

    cacheRunDirectory.clear();
    cacheGeneration = 0;
}

Result Plugin::Impl::load(const std::string& path) {
    if (path.empty()) {
        JST_ERROR("[PLUGIN] Cannot load plugin because path is empty.");
        return Result::ERROR;
    }

    if (!isCepPath(path)) {
        JST_ERROR("[PLUGIN] Plugin '{}' is not a .cep bundle.", path);
        return Result::ERROR;
    }

    const auto sourcePath = normalizePath(path);

    std::lock_guard<std::mutex> lifecycleGuard(lifecycleMutex);
    JST_CHECK(Registry::DrainStaticRegistrations());

    {
        std::lock_guard<std::mutex> guard(pluginsMutex);
        const auto duplicate = std::find_if(plugins.begin(), plugins.end(), [&](const auto& entry) {
            return entry.sourcePath == sourcePath;
        });
        if (duplicate != plugins.end()) {
            return Result::SUCCESS;
        }
    }

    Record plugin;
    JST_CHECK(loadPluginCopy(sourcePath, plugin));

    std::lock_guard<std::mutex> guard(pluginsMutex);
    plugins.push_back(std::move(plugin));
    return Result::SUCCESS;
}

Result Plugin::Impl::reload(const std::string& path) {
    if (path.empty()) {
        JST_ERROR("[PLUGIN] Cannot reload plugin because path is empty.");
        return Result::ERROR;
    }

    if (!isCepPath(path)) {
        JST_ERROR("[PLUGIN] Plugin '{}' is not a .cep bundle.", path);
        return Result::ERROR;
    }

    const auto sourcePath = normalizePath(path);

    std::lock_guard<std::mutex> lifecycleGuard(lifecycleMutex);
    JST_CHECK(Registry::DrainStaticRegistrations());

    Record plugin;
    bool loaded = false;
    {
        std::lock_guard<std::mutex> guard(pluginsMutex);
        const auto it = std::find_if(plugins.begin(), plugins.end(), [&](const auto& entry) {
            return entry.sourcePath == sourcePath;
        });
        if (it != plugins.end()) {
            plugin = std::move(*it);
            plugins.erase(it);
            loaded = true;
        }
    }

    Record newPlugin;
    if (loaded) {
        closePlugin(plugin, false);
    }

    if (loadPluginCopy(sourcePath, newPlugin) != Result::SUCCESS) {
        if (loaded) {
            Record restoredPlugin;
            if (loadPluginCopy(plugin.loadedPath, restoredPlugin) == Result::SUCCESS) {
                restoredPlugin.sourcePath = plugin.sourcePath;
                removeCachedPluginFiles(plugin);

                std::lock_guard<std::mutex> guard(pluginsMutex);
                plugins.push_back(std::move(restoredPlugin));
            } else {
                JST_ERROR("[PLUGIN] Failed to restore plugin '{}' after reload failure.", sourcePath);
                removeCachedPluginFiles(plugin);
            }
        }

        return Result::ERROR;
    }

    if (loaded) {
        removeCachedPluginFiles(plugin);
    }

    std::lock_guard<std::mutex> guard(pluginsMutex);
    plugins.push_back(std::move(newPlugin));
    return Result::SUCCESS;
}

std::vector<Plugin::Info> Plugin::Impl::list() {
    std::lock_guard<std::mutex> guard(pluginsMutex);

    std::vector<Plugin::Info> rows;
    rows.reserve(plugins.size());
    for (const auto& plugin : plugins) {
        rows.push_back(buildInfo(plugin));
    }

    return rows;
}

Result Plugin::Impl::loadPluginCopy(const std::string& sourcePath, Record& plugin) {
    if (!isCepPath(sourcePath)) {
        JST_ERROR("[PLUGIN] Plugin '{}' is not a .cep bundle.", sourcePath);
        return Result::ERROR;
    }

    const auto before = snapshotRegistry();

    std::string cachedPath;
    if (copyToCache(sourcePath, cachedPath) != Result::SUCCESS) {
        return Result::ERROR;
    }

    std::string extractedPath;
    std::vector<void*> handles;
    std::vector<std::string> loadedTargets;
    auto fail = [&]() {
        (void)Registry::DiscardStaticRegistrations();
        rollbackRegistrations(diffRegistrations(before, snapshotRegistry()));
        for (auto* handle : handles) {
            closeHandle(handle);
        }
        handles.clear();

        Record failedPlugin;
        failedPlugin.sourcePath = sourcePath;
        failedPlugin.loadedPath = cachedPath;
        failedPlugin.extractedPath = extractedPath;
        closePlugin(failedPlugin);
        return Result::ERROR;
    };

    if (extractToCache(cachedPath, extractedPath) != Result::SUCCESS) {
        return fail();
    }

    const auto bundlePath = std::filesystem::path(extractedPath);
    Manifest manifest;
    if (loadManifest(bundlePath, manifest) != Result::SUCCESS ||
        validateManifest(sourcePath, manifest) != Result::SUCCESS) {
        return fail();
    }

    bool loadedCompatibleTarget = false;
    for (const auto& target : manifest.targets) {
        if (!isCompatibleTarget(target)) {
            continue;
        }

        loadedCompatibleTarget = true;

        std::filesystem::path relativePath;
        if (!safeRelativePath(target.path, relativePath)) {
            return fail();
        }

        void* handle = nullptr;
        const auto libraryPath = Platform::PathToUtf8(bundlePath / relativePath);
        if (openPlugin(libraryPath, handle) != Result::SUCCESS) {
            if (handle != nullptr) {
                closeHandle(handle);
            }

            return fail();
        }

        handles.push_back(handle);

        if (Registry::DrainStaticRegistrations() != Result::SUCCESS) {
            return fail();
        }

        loadedTargets.push_back(target.path);
    }

    if (!loadedCompatibleTarget) {
        JST_ERROR("[PLUGIN] Plugin '{}' does not contain targets compatible with {}-{}.",
                  sourcePath,
                  currentSystem(),
                  currentArch());
        return fail();
    }

    if (registerExamples(bundlePath, manifest) != Result::SUCCESS) {
        return fail();
    }

    plugin.sourcePath = sourcePath;
    plugin.loadedPath = cachedPath;
    plugin.extractedPath = extractedPath;
    plugin.handles = std::move(handles);
    plugin.registrations = diffRegistrations(before, snapshotRegistry());
    plugin.manifest = std::move(manifest);
    plugin.loadedTargets = std::move(loadedTargets);
    return Result::SUCCESS;
}

Result Plugin::Impl::openPlugin(const std::string& path, void*& handle) {
    handle = nullptr;
    std::string error;

    try {
        handle = Platform::OpenDynamicLibrary(path, Platform::DynamicLibraryVisibility::Local, error);
    } catch (const Result& status) {
        JST_ERROR("[PLUGIN] Exception while loading plugin '{}': {}", path, status);
        return Result::ERROR;
    } catch (const std::exception& e) {
        JST_ERROR("[PLUGIN] Exception while loading plugin '{}': {}", path, e.what());
        return Result::ERROR;
    } catch (...) {
        JST_ERROR("[PLUGIN] Unknown exception while loading plugin '{}'.", path);
        return Result::ERROR;
    }

    if (handle == nullptr) {
        JST_ERROR("[PLUGIN] Failed to load plugin '{}': {}", path, error);
        return Result::ERROR;
    }

    JST_CHECK(validatePluginAbi(path, handle));
    return Result::SUCCESS;
}

Result Plugin::Impl::validatePluginAbi(const std::string& path, void* handle) {
    std::string error;
    const auto abi = loadAbi(handle, error);
    if (abi == nullptr) {
        JST_ERROR("[PLUGIN] Plugin '{}' does not export '{}': {}.",
                  path,
                  JETSTREAM_PLUGIN_ABI_SYMBOL,
                  error);
        return Result::ERROR;
    }

    if (abi->magic != JETSTREAM_PLUGIN_ABI_MAGIC) {
        JST_ERROR("[PLUGIN] Plugin '{}' has invalid plugin ABI magic.", path);
        return Result::ERROR;
    }

    if (abi->size != sizeof(JetstreamPluginAbi)) {
        JST_ERROR("[PLUGIN] Plugin '{}' reports an unsupported plugin ABI size ({} != {}).",
                  path,
                  abi->size,
                  sizeof(JetstreamPluginAbi));
        return Result::ERROR;
    }

    if (abi->abi_version != JETSTREAM_PLUGIN_ABI_VERSION) {
        JST_ERROR("[PLUGIN] Plugin '{}' has plugin ABI version {}, expected {}.",
                  path,
                  abi->abi_version,
                  JETSTREAM_PLUGIN_ABI_VERSION);
        return Result::ERROR;
    }

    JST_TRACE("[PLUGIN] Plugin '{}' ABI validated.", path);
    return Result::SUCCESS;
}

Result Plugin::Impl::registerExamples(const std::filesystem::path& bundlePath, const Manifest& manifest) {
    for (const auto& example : manifest.examples) {
        std::filesystem::path relativePath;
        if (!safeRelativePath(example.path, relativePath)) {
            JST_ERROR("[PLUGIN] Invalid bundled example path '{}'.", example.path);
            return Result::ERROR;
        }

        const auto examplePath = bundlePath / relativePath;
        std::string content;
        JST_CHECK(readTextFile(examplePath, content));

        Registry::FlowgraphRegistration record;
        record.key = examplePath.stem().string();
        if (record.key.empty()) {
            record.key = relativePath.string();
        }
        record.title = record.key;
        record.content = content;

        Parser::Map data;
        auto result = Parser::YamlDecode(record.content, data);
        if (result != Result::SUCCESS) {
            return result;
        }

        result = record.deserialize(data);
        if (result != Result::SUCCESS) {
            return result;
        }

        if (record.title.empty()) {
            record.title = record.key;
        }

        JST_CHECK(Registry::RegisterFlowgraph(record.key, record));
    }

    return Result::SUCCESS;
}

void Plugin::Impl::closePlugin(Record& plugin, bool removeCachedFile) {
    rollbackRegistrations(plugin.registrations);
    for (auto it = plugin.handles.rbegin(); it != plugin.handles.rend(); ++it) {
        closeHandle(*it);
    }
    plugin.handles.clear();

    if (removeCachedFile) {
        removeCachedPluginFiles(plugin);
    }
}

void Plugin::Impl::removeCachedPluginFiles(Record& plugin) {
    std::error_code ec;

    if (!plugin.extractedPath.empty()) {
        (void)std::filesystem::remove_all(std::filesystem::path(plugin.extractedPath), ec);
        if (ec) {
            JST_WARN("[PLUGIN] Failed to remove extracted plugin '{}'.", plugin.extractedPath);
        }

        plugin.extractedPath.clear();
    }

    if (!plugin.loadedPath.empty()) {
        ec.clear();
        (void)std::filesystem::remove(std::filesystem::path(plugin.loadedPath), ec);
        if (ec) {
            JST_WARN("[PLUGIN] Failed to remove cached plugin '{}'.", plugin.loadedPath);
        }

        plugin.loadedPath.clear();
    }
}

void Plugin::Impl::cleanup() {
    std::lock_guard<std::mutex> lifecycleGuard(lifecycleMutex);

    std::vector<Record> records;
    {
        std::lock_guard<std::mutex> guard(pluginsMutex);
        records.swap(plugins);
    }

    for (auto& plugin : records) {
        closePlugin(plugin);
    }

    cleanupCache();
}

Plugin::Impl& Plugin::plugin() {
    static Impl impl;
    return impl;
}

Result Plugin::Load(const std::string& path) {
    JST_CHECK(plugin().load(path));
    return Result::SUCCESS;
}

Result Plugin::Reload(const std::string& path) {
    JST_CHECK(plugin().reload(path));
    return Result::SUCCESS;
}

std::vector<Plugin::Info> Plugin::List() {
    return plugin().list();
}

}  // namespace Jetstream
