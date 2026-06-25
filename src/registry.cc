#include "jetstream/registry.hh"

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <functional>
#include <mutex>
#include <utility>

namespace Jetstream {

namespace {

thread_local std::vector<std::function<Result()>>* target = nullptr;

class ActiveQueueScope {
 public:
    explicit ActiveQueueScope(std::vector<std::function<Result()>>* queue) : previousQueue(target) {
        target = queue;
    }

    ~ActiveQueueScope() {
        target = previousQueue;
    }

 private:
    std::vector<std::function<Result()>>* previousQueue;
};

}  // namespace

struct Registry::Impl {
    Result queueStaticRegistration(std::function<Result()> callback);
    Result drainStaticRegistrations();
    Result discardStaticRegistrations();

    Result registerModule(const std::string& type,
                          DeviceType device,
                          RuntimeType runtime,
                          const ProviderType& provider,
                          Registry::ModuleFactory factory);
    Result registerBlock(const std::string& type,
                         const std::string& domain,
                         const std::string& title,
                         const std::string& summary,
                         const std::string& description,
                         Registry::BlockFactory factory);
    Result registerFlowgraph(const std::string& key,
                             const Registry::FlowgraphRegistration& metadata);
    Result registerBenchmark(const std::string& moduleType,
                             Registry::BenchmarkFactory factory,
                             const void* owner);

    Result unregisterModule(const std::string& type,
                            DeviceType device,
                            RuntimeType runtime,
                            const ProviderType& provider);
    Result unregisterBlock(const std::string& type);
    Result unregisterFlowgraph(const std::string& key);
    Result unregisterBenchmark(const std::string& moduleType,
                               const void* owner);

    std::vector<Registry::ModuleRegistration> listModules(const std::string& type,
                                                          std::optional<DeviceType> device,
                                                          std::optional<RuntimeType> runtime,
                                                          const ProviderType& provider);
    std::vector<Registry::FlowgraphRegistration> listFlowgraphs(const std::string& key);
    std::vector<Registry::BlockRegistration> listBlocks(const std::string& type);
    std::vector<Registry::BenchmarkRegistration> listBenchmarks(const std::string& moduleType);

    Result buildModule(const std::string& type,
                       const DeviceType& device,
                       const RuntimeType& runtime,
                       const ProviderType& provider,
                       std::shared_ptr<Module>& module,
                       const std::shared_ptr<Flowgraph::Environment>& environment,
                       const std::shared_ptr<Flowgraph::View>& view);
    Result buildBlock(const std::string& type, std::shared_ptr<Block>& block);

    std::mutex pendingRegistrationMutex;
    std::recursive_mutex registrationDrainMutex;
    std::vector<std::function<Result()>> pendingRegistrations;

    std::mutex registrationsMutex;
    std::vector<Registry::ModuleRegistration> modules;
    std::vector<Registry::BlockRegistration> blocks;
    std::vector<Registry::FlowgraphRegistration> flowgraphs;
    std::vector<Registry::BenchmarkRegistration> benchmarks;
};

Registry::Impl& Registry::registry() {
    static Impl impl;
    return impl;
}

Result Registry::Impl::queueStaticRegistration(std::function<Result()> callback) {
    if (!callback) {
        JST_ERROR("[REGISTRY] Cannot queue an empty static registration.");
        return Result::ERROR;
    }

    if (target != nullptr) {
        auto* queue = target;
        queue->push_back(std::move(callback));
        return Result::SUCCESS;
    }

    std::lock_guard<std::mutex> guard(pendingRegistrationMutex);
    pendingRegistrations.push_back(std::move(callback));
    return Result::SUCCESS;
}

Result Registry::Impl::drainStaticRegistrations() {
    std::lock_guard<std::recursive_mutex> guard(registrationDrainMutex);

    std::vector<std::function<Result()>> callbacks;
    {
        std::lock_guard<std::mutex> pendingGuard(pendingRegistrationMutex);
        callbacks.swap(pendingRegistrations);
    }

    ActiveQueueScope queueScope(&callbacks);

    while (!callbacks.empty()) {
        auto pending = std::move(callbacks);
        callbacks.clear();

        for (const auto& callback : pending) {
            try {
                if (callback() != Result::SUCCESS) {
                    return Result::ERROR;
                }
            } catch (const Result& status) {
                JST_ERROR("[REGISTRY] Exception while processing deferred registration: {}", status);
                return Result::ERROR;
            } catch (const std::exception& e) {
                JST_ERROR("[REGISTRY] Exception while processing deferred registration: {}", e.what());
                return Result::ERROR;
            } catch (...) {
                JST_ERROR("[REGISTRY] Unknown exception while processing deferred registration.");
                return Result::ERROR;
            }
        }
    }

    return Result::SUCCESS;
}

Result Registry::Impl::discardStaticRegistrations() {
    std::lock_guard<std::mutex> guard(pendingRegistrationMutex);
    pendingRegistrations.clear();
    return Result::SUCCESS;
}

Result Registry::Impl::registerModule(const std::string& type,
                                      DeviceType device,
                                      RuntimeType runtime,
                                      const ProviderType& provider,
                                      Registry::ModuleFactory factory) {
    JST_TRACE("[REGISTRY] Registering module [Type: {}, Runtime: {}, Device: {}, Provider: {}]", type, runtime, device, provider);

    if (type.empty()) {
        JST_ERROR("[REGISTRY] Empty type for module registration.");
        return Result::ERROR;
    }
    if (device == DeviceType::None) {
        JST_ERROR("[REGISTRY] Invalid device type ('{}') for module '{}'.", device, type);
        return Result::ERROR;
    }
    if (runtime == RuntimeType::NONE) {
        JST_ERROR("[REGISTRY] Invalid runtime type ('{}') for module '{}'.", runtime, type);
        return Result::ERROR;
    }
    if (provider.empty()) {
        JST_ERROR("[REGISTRY] Empty provider for module '{}'.", type);
        return Result::ERROR;
    }

    std::lock_guard<std::mutex> guard(registrationsMutex);

    const auto duplicate = std::find_if(modules.begin(), modules.end(), [&](const auto& entry) {
        return entry.type == type &&
               entry.device == device &&
               entry.runtime == runtime &&
               entry.provider == provider;
    });

    if (duplicate != modules.end()) {
        JST_ERROR("[REGISTRY] Module already registered [Type: {}, Device: {}, Runtime: {}, Provider: {}].", type, device, runtime, provider);
        return Result::ERROR;
    }

    modules.push_back({type, device, runtime, provider, std::move(factory)});
    return Result::SUCCESS;
}

Result Registry::Impl::registerBlock(const std::string& type,
                                     const std::string& domain,
                                     const std::string& title,
                                     const std::string& summary,
                                     const std::string& description,
                                     Registry::BlockFactory factory) {
    JST_TRACE("[REGISTRY] Registering block [Type: {}, Domain: {}]", type, domain);

    if (type.empty()) {
        JST_ERROR("[REGISTRY] Empty type for block registration.");
        return Result::ERROR;
    }
    if (domain.empty()) {
        JST_ERROR("[REGISTRY] Empty domain for block '{}'.", type);
        return Result::ERROR;
    }

    std::lock_guard<std::mutex> guard(registrationsMutex);

    const auto duplicate = std::find_if(blocks.begin(), blocks.end(), [&](const auto& entry) {
        return entry.type == type;
    });

    if (duplicate != blocks.end()) {
        if (duplicate->domain == domain &&
            duplicate->title == title &&
            duplicate->summary == summary &&
            duplicate->description == description) {
            return Result::SUCCESS;
        }

        JST_ERROR("[REGISTRY] Block already registered [Type: {}].", type);
        return Result::ERROR;
    }

    blocks.push_back({type, domain, title, summary, description, std::move(factory)});
    return Result::SUCCESS;
}

Result Registry::Impl::registerFlowgraph(const std::string& key,
                                         const Registry::FlowgraphRegistration& metadata) {
    JST_TRACE("[REGISTRY] Registering flowgraph [Key: {}]", key);

    if (key.empty()) {
        JST_ERROR("[REGISTRY] Empty key for flowgraph registration.");
        return Result::ERROR;
    }

    std::lock_guard<std::mutex> guard(registrationsMutex);

    const auto duplicate = std::find_if(flowgraphs.begin(), flowgraphs.end(), [&](const auto& entry) {
        return entry.key == key;
    });

    if (duplicate != flowgraphs.end()) {
        JST_ERROR("[REGISTRY] Flowgraph already registered [Key: {}].", key);
        return Result::ERROR;
    }

    flowgraphs.push_back(metadata);
    return Result::SUCCESS;
}

Result Registry::Impl::registerBenchmark(const std::string& moduleType,
                                         Registry::BenchmarkFactory factory,
                                         const void* owner) {
    JST_TRACE("[REGISTRY] Registering benchmark [Module: {}]", moduleType);

    if (moduleType.empty()) {
        JST_ERROR("[REGISTRY] Empty module type for benchmark spec registration.");
        return Result::ERROR;
    }
    if (!factory) {
        JST_ERROR("[REGISTRY] Empty benchmark factory for module '{}'.", moduleType);
        return Result::ERROR;
    }
    if (owner == nullptr) {
        JST_ERROR("[REGISTRY] Empty benchmark spec owner for module '{}'.", moduleType);
        return Result::ERROR;
    }

    std::lock_guard<std::mutex> guard(registrationsMutex);

    const auto duplicate = std::find_if(benchmarks.begin(), benchmarks.end(), [&](const auto& entry) {
        return entry.moduleType == moduleType && entry.owner == owner;
    });

    if (duplicate != benchmarks.end()) {
        JST_ERROR("[REGISTRY] Benchmark already registered [Module: {}].", moduleType);
        return Result::ERROR;
    }

    benchmarks.push_back({moduleType, owner, std::move(factory)});
    return Result::SUCCESS;
}

Result Registry::Impl::unregisterModule(const std::string& type,
                                        DeviceType device,
                                        RuntimeType runtime,
                                        const ProviderType& provider) {
    std::lock_guard<std::mutex> guard(registrationsMutex);
    const auto size = modules.size();
    modules.erase(
        std::remove_if(modules.begin(), modules.end(), [&](const auto& entry) {
            return entry.type == type &&
                   entry.device == device &&
                   entry.runtime == runtime &&
                   entry.provider == provider;
        }),
        modules.end());
    return modules.size() == size ? Result::ERROR : Result::SUCCESS;
}

Result Registry::Impl::unregisterBlock(const std::string& type) {
    std::lock_guard<std::mutex> guard(registrationsMutex);
    const auto size = blocks.size();
    blocks.erase(
        std::remove_if(blocks.begin(), blocks.end(), [&](const auto& entry) {
            return entry.type == type;
        }),
        blocks.end());
    return blocks.size() == size ? Result::ERROR : Result::SUCCESS;
}

Result Registry::Impl::unregisterFlowgraph(const std::string& key) {
    std::lock_guard<std::mutex> guard(registrationsMutex);
    const auto size = flowgraphs.size();
    flowgraphs.erase(
        std::remove_if(flowgraphs.begin(), flowgraphs.end(), [&](const auto& entry) {
            return entry.key == key;
        }),
        flowgraphs.end());
    return flowgraphs.size() == size ? Result::ERROR : Result::SUCCESS;
}

Result Registry::Impl::unregisterBenchmark(const std::string& moduleType,
                                           const void* owner) {
    std::lock_guard<std::mutex> guard(registrationsMutex);
    const auto size = benchmarks.size();
    benchmarks.erase(
        std::remove_if(benchmarks.begin(), benchmarks.end(), [&](const auto& entry) {
            return entry.moduleType == moduleType && entry.owner == owner;
        }),
        benchmarks.end());
    return benchmarks.size() == size ? Result::ERROR : Result::SUCCESS;
}

std::vector<Registry::ModuleRegistration> Registry::Impl::listModules(const std::string& type,
                                                                      std::optional<DeviceType> device,
                                                                      std::optional<RuntimeType> runtime,
                                                                      const ProviderType& provider) {
    JST_TRACE("[REGISTRY] Listing modules.");
    std::lock_guard<std::mutex> guard(registrationsMutex);

    std::vector<Registry::ModuleRegistration> filtered;
    filtered.reserve(modules.size());

    for (const auto& module : modules) {
        if (!type.empty() && module.type != type) {
            continue;
        }

        if (device && module.device != *device) {
            continue;
        }

        if (runtime && module.runtime != *runtime) {
            continue;
        }

        if (!provider.empty() && module.provider != provider) {
            continue;
        }

        filtered.push_back(module);
    }

    return filtered;
}

std::vector<Registry::FlowgraphRegistration> Registry::Impl::listFlowgraphs(const std::string& key) {
    JST_TRACE("[REGISTRY] Listing flowgraphs.");
    std::lock_guard<std::mutex> guard(registrationsMutex);

    std::vector<Registry::FlowgraphRegistration> filtered;
    filtered.reserve(flowgraphs.size());

    for (const auto& flowgraph : flowgraphs) {
        if (!key.empty() && flowgraph.key != key) {
            continue;
        }

        filtered.push_back(flowgraph);
    }

    return filtered;
}

std::vector<Registry::BlockRegistration> Registry::Impl::listBlocks(const std::string& type) {
    JST_TRACE("[REGISTRY] Listing blocks.");
    std::lock_guard<std::mutex> guard(registrationsMutex);

    std::vector<Registry::BlockRegistration> filtered;
    filtered.reserve(blocks.size());

    for (const auto& block : blocks) {
        if (!type.empty() && block.type != type) {
            continue;
        }

        filtered.push_back(block);
    }

    return filtered;
}

std::vector<Registry::BenchmarkRegistration> Registry::Impl::listBenchmarks(const std::string& moduleType) {
    JST_TRACE("[REGISTRY] Listing benchmarks.");
    std::lock_guard<std::mutex> guard(registrationsMutex);

    std::vector<Registry::BenchmarkRegistration> filtered;
    filtered.reserve(benchmarks.size());

    for (const auto& benchmark : benchmarks) {
        if (!moduleType.empty() && benchmark.moduleType != moduleType) {
            continue;
        }

        filtered.push_back(benchmark);
    }

    return filtered;
}

Result Registry::Impl::buildModule(const std::string& type,
                                   const DeviceType& device,
                                   const RuntimeType& runtime,
                                   const ProviderType& provider,
                                   std::shared_ptr<Module>& module,
                                   const std::shared_ptr<Flowgraph::Environment>& environment,
                                   const std::shared_ptr<Flowgraph::View>& view) {
    JST_TRACE("[REGISTRY] Creating module [Type: {}, Device: {}, Runtime: {}, Provider: {}].", type, device, runtime, provider);

    if (device == DeviceType::None) {
        JST_ERROR("[REGISTRY] Invalid device type ('{}') for module '{}'.", device, type);
        return Result::ERROR;
    }
    if (runtime == RuntimeType::NONE) {
        JST_ERROR("[REGISTRY] Invalid runtime type ('{}') for module '{}'.", runtime, type);
        return Result::ERROR;
    }
    if (provider.empty()) {
        JST_ERROR("[REGISTRY] Empty provider for module '{}'.", type);
        return Result::ERROR;
    }

    std::lock_guard<std::mutex> guard(registrationsMutex);

    const auto it = std::find_if(modules.begin(), modules.end(), [&](const auto& entry) {
        return entry.type == type &&
               entry.device == device &&
               entry.runtime == runtime &&
               entry.provider == provider;
    });

    if (it != modules.end()) {
        module = it->factory(environment, view);
        return Result::SUCCESS;
    }

    JST_ERROR("[REGISTRY] Module not found [Type: {}, Device: {}, Runtime: {}, Provider: {}].", type, device, runtime, provider);
    return Result::ERROR;
}

Result Registry::Impl::buildBlock(const std::string& type, std::shared_ptr<Block>& block) {
    JST_TRACE("[REGISTRY] Creating block [Type: {}]", type);
    std::lock_guard<std::mutex> guard(registrationsMutex);

    const auto it = std::find_if(blocks.begin(), blocks.end(), [&](const auto& entry) {
        return entry.type == type;
    });

    if (it != blocks.end()) {
        block = it->factory();
        return Result::SUCCESS;
    }

    JST_ERROR("[REGISTRY] Block not found [Type: {}]", type);
    return Result::ERROR;
}

Result Registry::QueueStaticRegistration(std::function<Result()> callback) {
    JST_CHECK(registry().queueStaticRegistration(std::move(callback)));
    return Result::SUCCESS;
}

Result Registry::DrainStaticRegistrations() {
    JST_CHECK(registry().drainStaticRegistrations());
    return Result::SUCCESS;
}

Result Registry::DiscardStaticRegistrations() {
    JST_CHECK(registry().discardStaticRegistrations());
    return Result::SUCCESS;
}

Result Registry::RegisterModule(const std::string& type,
                                DeviceType device,
                                RuntimeType runtime,
                                const ProviderType& provider,
                                ModuleFactory factory) {
    JST_CHECK(registry().registerModule(type, device, runtime, provider, std::move(factory)));
    return Result::SUCCESS;
}

Result Registry::RegisterBlock(const std::string& type,
                               const std::string& domain,
                               const std::string& title,
                               const std::string& summary,
                               const std::string& description,
                               BlockFactory factory) {
    JST_CHECK(registry().registerBlock(type, domain, title, summary, description, std::move(factory)));
    return Result::SUCCESS;
}

Result Registry::RegisterFlowgraph(const std::string& key,
                                   const FlowgraphRegistration& metadata) {
    JST_CHECK(registry().registerFlowgraph(key, metadata));
    return Result::SUCCESS;
}

Result Registry::RegisterBenchmark(const std::string& moduleType,
                                   Registry::BenchmarkFactory factory,
                                   const void* owner) {
    JST_CHECK(registry().registerBenchmark(moduleType, std::move(factory), owner));
    return Result::SUCCESS;
}

Result Registry::UnregisterModule(const std::string& type,
                                  DeviceType device,
                                  RuntimeType runtime,
                                  const ProviderType& provider) {
    JST_CHECK(registry().unregisterModule(type, device, runtime, provider));
    return Result::SUCCESS;
}

Result Registry::UnregisterBlock(const std::string& type) {
    JST_CHECK(registry().unregisterBlock(type));
    return Result::SUCCESS;
}

Result Registry::UnregisterFlowgraph(const std::string& key) {
    JST_CHECK(registry().unregisterFlowgraph(key));
    return Result::SUCCESS;
}

Result Registry::UnregisterBenchmark(const std::string& moduleType,
                                     const void* owner) {
    JST_CHECK(registry().unregisterBenchmark(moduleType, owner));
    return Result::SUCCESS;
}

std::vector<Registry::ModuleRegistration> Registry::ListAvailableModules(const std::string& type,
                                                                         std::optional<DeviceType> device,
                                                                         std::optional<RuntimeType> runtime,
                                                                         const ProviderType& provider) {
    if (registry().drainStaticRegistrations() != Result::SUCCESS) {
        std::abort();
    }
    return registry().listModules(type, device, runtime, provider);
}

std::vector<Registry::BlockRegistration> Registry::ListAvailableBlocks(const std::string& type) {
    if (registry().drainStaticRegistrations() != Result::SUCCESS) {
        std::abort();
    }
    return registry().listBlocks(type);
}

std::vector<Registry::FlowgraphRegistration> Registry::ListAvailableFlowgraphs(const std::string& key) {
    if (registry().drainStaticRegistrations() != Result::SUCCESS) {
        std::abort();
    }
    return registry().listFlowgraphs(key);
}

std::vector<Registry::BenchmarkRegistration> Registry::ListAvailableBenchmarks(const std::string& moduleType) {
    if (registry().drainStaticRegistrations() != Result::SUCCESS) {
        std::abort();
    }
    return registry().listBenchmarks(moduleType);
}

Result Registry::BuildModule(const std::string& type,
                             const DeviceType& device,
                             const RuntimeType& runtime,
                             const ProviderType& provider,
                             std::shared_ptr<Module>& module,
                             const std::shared_ptr<Flowgraph::Environment>& environment,
                             const std::shared_ptr<Flowgraph::View>& view) {
    JST_CHECK(registry().drainStaticRegistrations());
    JST_CHECK(registry().buildModule(type, device, runtime, provider, module, environment, view));
    return Result::SUCCESS;
}

Result Registry::BuildBlock(const std::string& type, std::shared_ptr<Block>& block) {
    JST_CHECK(registry().drainStaticRegistrations());
    JST_CHECK(registry().buildBlock(type, block));
    return Result::SUCCESS;
}

}  // namespace Jetstream
