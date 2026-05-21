#include "jetstream/plugin.hh"

#include "jetstream/logger.hh"
#include "jetstream/platform.hh"
#include "jetstream/registry.hh"
#include "jetstream/runtime.hh"

#include <algorithm>
#include <chrono>
#include <exception>
#include <filesystem>
#include <functional>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#if defined(JST_OS_WINDOWS)
#include <windows.h>
#elif !defined(JST_OS_BROWSER)
#include <dlfcn.h>
#endif

namespace Jetstream {

struct Plugin::Impl {
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
        void* handle = nullptr;
        Registrations registrations;
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

 private:
    static std::string pathToUtf8(const std::filesystem::path& path);
    static std::string normalizePath(const std::string& path);
    static std::string cacheFileName(const std::string& sourcePath, uint64_t generation);

    static void closeHandle(void* handle);
    static const JetstreamPluginAbi* loadAbi(void* handle);
    static uint64_t abiMask(DeviceType device);
    static uint64_t abiMask(RuntimeType runtime);
    static uint64_t availableDeviceMask();
    static uint64_t availableRuntimeMask();

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
    void sweepCache(const std::filesystem::path& runsDirectory);
    void cleanupCache();

    Result loadPluginCopy(const std::string& sourcePath, Record& plugin);
    Result openPlugin(const std::string& path, void*& handle);
    Result validatePluginAbi(const std::string& path, void* handle);
    void closePlugin(Record& plugin, bool removeCachedFile = true);
    void removeCachedPluginFile(Record& plugin);
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

std::string Plugin::Impl::pathToUtf8(const std::filesystem::path& path) {
    const auto utf8Path = path.u8string();

    std::string value;
    value.reserve(utf8Path.size());
    for (const auto ch : utf8Path) {
        value.push_back(static_cast<char>(ch));
    }

    return value;
}

std::string Plugin::Impl::normalizePath(const std::string& path) {
    std::error_code ec;
    const auto inputPath = std::filesystem::u8path(path);

    const auto canonicalPath = std::filesystem::weakly_canonical(inputPath, ec);
    if (!ec) {
        return pathToUtf8(canonicalPath);
    }

    ec.clear();
    const auto absolutePath = std::filesystem::absolute(inputPath, ec);
    if (!ec) {
        return pathToUtf8(absolutePath);
    }

    return path;
}

std::string Plugin::Impl::cacheFileName(const std::string& sourcePath, uint64_t generation) {
    auto filename = pathToUtf8(std::filesystem::u8path(sourcePath).filename());
    if (filename.empty()) {
        filename = "plugin";
    }

    const auto sourceHash = std::hash<std::string>{}(sourcePath);
    return std::to_string(sourceHash) + "-" + std::to_string(generation) + "-" + filename;
}

void Plugin::Impl::closeHandle(void* handle) {
    if (handle == nullptr) {
        return;
    }

#if defined(JST_OS_WINDOWS)
    FreeLibrary(reinterpret_cast<HMODULE>(handle));
#elif !defined(JST_OS_BROWSER)
    dlclose(handle);
#else
    (void)handle;
#endif
}

const JetstreamPluginAbi* Plugin::Impl::loadAbi(void* handle) {
#if defined(JST_OS_WINDOWS)
    return reinterpret_cast<const JetstreamPluginAbi*>(
        GetProcAddress(reinterpret_cast<HMODULE>(handle), JETSTREAM_PLUGIN_ABI_SYMBOL));
#elif !defined(JST_OS_BROWSER)
    dlerror();
    return reinterpret_cast<const JetstreamPluginAbi*>(dlsym(handle, JETSTREAM_PLUGIN_ABI_SYMBOL));
#else
    (void)handle;
    return nullptr;
#endif
}

uint64_t Plugin::Impl::abiMask(DeviceType device) {
    return static_cast<uint64_t>(static_cast<uint8_t>(device));
}

uint64_t Plugin::Impl::abiMask(RuntimeType runtime) {
    return static_cast<uint64_t>(static_cast<uint8_t>(runtime));
}

uint64_t Plugin::Impl::availableDeviceMask() {
    uint64_t mask = 0;

#if defined(JETSTREAM_BACKEND_CPU_AVAILABLE)
    mask |= abiMask(DeviceType::CPU);
#endif
#if defined(JETSTREAM_BACKEND_CUDA_AVAILABLE)
    mask |= abiMask(DeviceType::CUDA);
#endif
#if defined(JETSTREAM_BACKEND_METAL_AVAILABLE)
    mask |= abiMask(DeviceType::Metal);
#endif
#if defined(JETSTREAM_BACKEND_VULKAN_AVAILABLE)
    mask |= abiMask(DeviceType::Vulkan);
#endif
#if defined(JETSTREAM_BACKEND_WEBGPU_AVAILABLE)
    mask |= abiMask(DeviceType::WebGPU);
#endif

    return mask;
}

uint64_t Plugin::Impl::availableRuntimeMask() {
    uint64_t mask = abiMask(RuntimeType::NATIVE);

#if defined(JETSTREAM_LOADER_MLIR_AVAILABLE)
    mask |= abiMask(RuntimeType::MLIR);
#endif

    return mask;
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
    std::filesystem::copy_file(std::filesystem::u8path(sourcePath),
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

    cachedPath = pathToUtf8(destination);
    return Result::SUCCESS;
}

Result Plugin::Impl::ensureCacheReady() {
    if (!cacheRunDirectory.empty()) {
        return Result::SUCCESS;
    }

    std::string cachePath;
    JST_CHECK(Platform::CachePath(cachePath));

    const auto cacheRoot = std::filesystem::u8path(cachePath) / "registry-plugins";
    const auto runsDirectory = cacheRoot / "runs";

    std::error_code ec;
    std::filesystem::create_directories(runsDirectory, ec);
    if (ec) {
        JST_ERROR("[PLUGIN] Failed to create plugin cache directory '{}'.", runsDirectory.string());
        return Result::ERROR;
    }

    Platform::FileLock maintenanceLock;
    JST_CHECK(maintenanceLock.acquire(pathToUtf8(cacheRoot / "maintenance.lock")));

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
            if (cacheOwnerLock.acquire(pathToUtf8(candidate / "owner.lock")) != Result::SUCCESS) {
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
        const auto lockResult = staleOwnerLock.acquire(pathToUtf8(entry.path() / "owner.lock"), false);
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
                removeCachedPluginFile(plugin);

                std::lock_guard<std::mutex> guard(pluginsMutex);
                plugins.push_back(std::move(restoredPlugin));
            } else {
                JST_ERROR("[PLUGIN] Failed to restore plugin '{}' after reload failure.", sourcePath);
                removeCachedPluginFile(plugin);
            }
        }

        return Result::ERROR;
    }

    if (loaded) {
        removeCachedPluginFile(plugin);
    }

    std::lock_guard<std::mutex> guard(pluginsMutex);
    plugins.push_back(std::move(newPlugin));
    return Result::SUCCESS;
}

Result Plugin::Impl::loadPluginCopy(const std::string& sourcePath, Record& plugin) {
    const auto before = snapshotRegistry();

    std::string cachedPath;
    if (copyToCache(sourcePath, cachedPath) != Result::SUCCESS) {
        return Result::ERROR;
    }

    void* handle = nullptr;
    if (openPlugin(cachedPath, handle) != Result::SUCCESS) {
        (void)Registry::DiscardStaticRegistrations();
        rollbackRegistrations(diffRegistrations(before, snapshotRegistry()));
        Record failedPlugin;
        failedPlugin.sourcePath = sourcePath;
        failedPlugin.loadedPath = cachedPath;
        failedPlugin.handle = handle;
        closePlugin(failedPlugin);
        return Result::ERROR;
    }

    if (Registry::DrainStaticRegistrations() != Result::SUCCESS) {
        (void)Registry::DiscardStaticRegistrations();
        rollbackRegistrations(diffRegistrations(before, snapshotRegistry()));
        closeHandle(handle);
        handle = nullptr;

        Record failedPlugin;
        failedPlugin.sourcePath = sourcePath;
        failedPlugin.loadedPath = cachedPath;
        closePlugin(failedPlugin);
        return Result::ERROR;
    }

    plugin.sourcePath = sourcePath;
    plugin.loadedPath = cachedPath;
    plugin.handle = handle;
    plugin.registrations = diffRegistrations(before, snapshotRegistry());
    return Result::SUCCESS;
}

Result Plugin::Impl::openPlugin(const std::string& path, void*& handle) {
    handle = nullptr;

    try {
#if defined(JST_OS_WINDOWS)
        handle = reinterpret_cast<void*>(LoadLibraryA(path.c_str()));
#elif defined(JST_OS_BROWSER)
        JST_ERROR("[PLUGIN] Plugins are not supported in this platform.");
        return Result::ERROR;
#else
        dlerror();
        handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
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

#if defined(JST_OS_WINDOWS)
    if (handle == nullptr) {
        JST_ERROR("[PLUGIN] Failed to load plugin '{}'.", path);
        return Result::ERROR;
    }
#elif !defined(JST_OS_BROWSER)
    if (handle == nullptr) {
        const char* error = dlerror();
        JST_ERROR("[PLUGIN] Failed to load plugin '{}': {}", path, error != nullptr ? error : "unknown error");
        return Result::ERROR;
    }
#endif

    JST_CHECK(validatePluginAbi(path, handle));
    return Result::SUCCESS;
}

Result Plugin::Impl::validatePluginAbi(const std::string& path, void* handle) {
    const auto abi = loadAbi(handle);
    if (abi == nullptr) {
#if defined(JST_OS_WINDOWS)
        JST_ERROR("[PLUGIN] Plugin '{}' does not export '{}'.", path, JETSTREAM_PLUGIN_ABI_SYMBOL);
#elif !defined(JST_OS_BROWSER)
        const char* error = dlerror();
        JST_ERROR("[PLUGIN] Plugin '{}' does not export '{}': {}.",
                  path,
                  JETSTREAM_PLUGIN_ABI_SYMBOL,
                  error != nullptr ? error : "unknown error");
#endif
        return Result::ERROR;
    }

    if (abi->magic != JETSTREAM_PLUGIN_ABI_MAGIC) {
        JST_ERROR("[PLUGIN] Plugin '{}' has invalid plugin ABI magic.", path);
        return Result::ERROR;
    }

    if (abi->size < sizeof(JetstreamPluginAbi)) {
        JST_ERROR("[PLUGIN] Plugin '{}' reports an unsupported plugin ABI size ({} < {}).",
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

    if (abi->min_jetstream_version > JETSTREAM_VERSION_CURRENT) {
        JST_ERROR("[PLUGIN] Plugin '{}' requires Jetstream version {}, current version is {}.",
                  path,
                  abi->min_jetstream_version,
                  JETSTREAM_VERSION_CURRENT);
        return Result::ERROR;
    }

    const auto missingDevices = abi->required_devices & ~availableDeviceMask();
    if (missingDevices != 0) {
        JST_ERROR("[PLUGIN] Plugin '{}' requires unavailable devices [Mask: {}].", path, missingDevices);
        return Result::ERROR;
    }

    const auto missingRuntimes = abi->required_runtimes & ~availableRuntimeMask();
    if (missingRuntimes != 0) {
        JST_ERROR("[PLUGIN] Plugin '{}' requires unavailable runtimes [Mask: {}].", path, missingRuntimes);
        return Result::ERROR;
    }

    JST_TRACE("[PLUGIN] Plugin '{}' ABI validated [Name: {}, Version: {}].",
              path,
              abi->name != nullptr ? abi->name : "unknown",
              abi->version != nullptr ? abi->version : "unknown");
    return Result::SUCCESS;
}

void Plugin::Impl::closePlugin(Record& plugin, bool removeCachedFile) {
    rollbackRegistrations(plugin.registrations);
    closeHandle(plugin.handle);
    plugin.handle = nullptr;

    if (removeCachedFile) {
        removeCachedPluginFile(plugin);
    }
}

void Plugin::Impl::removeCachedPluginFile(Record& plugin) {
    if (plugin.loadedPath.empty()) {
        return;
    }

    std::error_code ec;
    (void)std::filesystem::remove(std::filesystem::u8path(plugin.loadedPath), ec);
    if (ec) {
        JST_WARN("[PLUGIN] Failed to remove cached plugin '{}'.", plugin.loadedPath);
    }

    plugin.loadedPath.clear();
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

}  // namespace Jetstream
