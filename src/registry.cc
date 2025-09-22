#include "jetstream/registry.hh"

#include <algorithm>
#include <cstddef>
#include <mutex>

namespace Jetstream {

struct Registry::Impl {
    Result registerModule(const std::string& type,
                          DeviceType device,
                          RuntimeType runtime,
                          const ProviderType& provider,
                          ModuleFactory factory);
    Result registerBlock(const std::string& type,
                         const std::string& title,
                         const std::string& summary,
                         const std::string& description,
                         BlockFactory factory);
    Result registerFlowgraph(const std::string& key,
                             const FlowgraphRegistration& metadata);
    std::vector<ModuleRegistration> listModules(const std::string& type,
                                                std::optional<DeviceType> device,
                                                std::optional<RuntimeType> runtime,
                                                const ProviderType& provider);
    std::vector<FlowgraphRegistration> listFlowgraphs(const std::string& key);
    std::vector<BlockRegistration> listBlocks(const std::string& type);
    Result buildModule(const std::string& type,
                       const DeviceType& device,
                       const RuntimeType& runtime,
                       const ProviderType& provider,
                       std::shared_ptr<Module>& module);
    Result buildBlock(const std::string& type, std::shared_ptr<Block>& block);

    std::mutex mutex;
    std::vector<ModuleRegistration> modules;
    std::vector<BlockRegistration> blocks;
    std::vector<FlowgraphRegistration> flowgraphs;
};

Registry::Impl& Registry::registry() {
    static Impl impl;
    return impl;
}

Result Registry::Impl::registerModule(const std::string& type,
                                      DeviceType device,
                                      RuntimeType runtime,
                                      const ProviderType& provider,
                                      ModuleFactory factory) {
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

    std::lock_guard<std::mutex> guard(mutex);

    auto duplicate = std::find_if(modules.begin(), modules.end(), [&](const ModuleRegistration& entry) {
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
                                     const std::string& title,
                                     const std::string& summary,
                                     const std::string& description,
                                     BlockFactory factory) {
    JST_TRACE("[REGISTRY] Registering block [Type: {}]", type);

    if (type.empty()) {
        JST_ERROR("[REGISTRY] Empty type for block registration.");
        return Result::ERROR;
    }

    std::lock_guard<std::mutex> guard(mutex);

    auto duplicate = std::find_if(blocks.begin(), blocks.end(), [&](const BlockRegistration& entry) {
        return entry.type == type;
    });

    if (duplicate != blocks.end()) {
        JST_ERROR("[REGISTRY] Block already registered [Type: {}].", type);
        return Result::ERROR;
    }

    blocks.push_back({type, title, summary, description, factory});
    return Result::SUCCESS;
}

Result Registry::Impl::registerFlowgraph(const std::string& key,
                                         const FlowgraphRegistration& metadata) {
    JST_TRACE("[REGISTRY] Registering flowgraph [Key: {}]", key);

    if (key.empty()) {
        JST_ERROR("[REGISTRY] Empty key for flowgraph registration.");
        return Result::ERROR;
    }

    std::lock_guard<std::mutex> guard(mutex);

    auto duplicate = std::find_if(flowgraphs.begin(), flowgraphs.end(), [&](const FlowgraphRegistration& entry) {
        return entry.key == key;
    });

    if (duplicate != flowgraphs.end()) {
        JST_ERROR("[REGISTRY] Flowgraph already registered [Key: {}].", key);
        return Result::ERROR;
    }

    flowgraphs.push_back(metadata);
    return Result::SUCCESS;
}

std::vector<Registry::ModuleRegistration> Registry::Impl::listModules(const std::string& type,
                                                                      std::optional<DeviceType> device,
                                                                      std::optional<RuntimeType> runtime,
                                                                      const ProviderType& provider) {
    JST_TRACE("[REGISTRY] Listing modules.");
    std::lock_guard<std::mutex> guard(mutex);

    std::vector<ModuleRegistration> filtered;
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
    std::lock_guard<std::mutex> guard(mutex);

    std::vector<FlowgraphRegistration> filtered;
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
    std::lock_guard<std::mutex> guard(mutex);

    std::vector<BlockRegistration> filtered;
    filtered.reserve(blocks.size());

    for (const auto& block : blocks) {
        if (!type.empty() && block.type != type) {
            continue;
        }

        filtered.push_back(block);
    }

    return filtered;
}

Result Registry::Impl::buildModule(const std::string& type,
                                   const DeviceType& device,
                                   const RuntimeType& runtime,
                                   const ProviderType& provider,
                                   std::shared_ptr<Module>& module) {
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

    std::lock_guard<std::mutex> guard(mutex);

    auto it = std::find_if(modules.begin(), modules.end(), [&](const ModuleRegistration& entry) {
        return entry.type == type &&
               entry.device == device &&
               entry.runtime == runtime &&
               entry.provider == provider;
    });

    if (it != modules.end()) {
        module = it->factory();
        return Result::SUCCESS;
    }

    JST_ERROR("[REGISTRY] Module not found [Type: {}, Device: {}, Runtime: {}, Provider: {}].", type, device, runtime, provider);
    return Result::ERROR;
}

Result Registry::Impl::buildBlock(const std::string& type, std::shared_ptr<Block>& block) {
    JST_TRACE("[REGISTRY] Creating block [Type: {}]", type);
    std::lock_guard<std::mutex> guard(mutex);

    auto it = std::find_if(blocks.begin(), blocks.end(), [&](const BlockRegistration& entry) {
        return entry.type == type;
    });

    if (it != blocks.end()) {
        block = it->factory();
        return Result::SUCCESS;
    }

    JST_ERROR("[REGISTRY] Block not found [Type: {}]", type);
    return Result::ERROR;
}

Result Registry::RegisterModule(const std::string& type,
                                DeviceType device,
                                RuntimeType runtime,
                                const ProviderType& provider,
                                ModuleFactory factory) {
    return registry().registerModule(type, device, runtime, provider, std::move(factory));
}

Result Registry::RegisterBlock(const std::string& type,
                               const std::string& title,
                               const std::string& summary,
                               const std::string& description,
                               BlockFactory factory) {
    return registry().registerBlock(type, title, summary, description, std::move(factory));
}

Result Registry::RegisterFlowgraph(const std::string& key,
                                   const FlowgraphRegistration& metadata) {
    return registry().registerFlowgraph(key, metadata);
}

std::vector<Registry::ModuleRegistration> Registry::ListAvailableModules(const std::string& type,
                                                                         std::optional<DeviceType> device,
                                                                         std::optional<RuntimeType> runtime,
                                                                         const ProviderType& provider) {
    return registry().listModules(type, device, runtime, provider);
}

std::vector<Registry::BlockRegistration> Registry::ListAvailableBlocks(const std::string& type) {
    return registry().listBlocks(type);
}

std::vector<Registry::FlowgraphRegistration> Registry::ListAvailableFlowgraphs(const std::string& key) {
    return registry().listFlowgraphs(key);
}

Result Registry::BuildModule(const std::string& type,
                             const DeviceType& device,
                             const RuntimeType& runtime,
                             const ProviderType& provider,
                             std::shared_ptr<Module>& module) {
    return registry().buildModule(type, device, runtime, provider, module);
}

Result Registry::BuildBlock(const std::string& type, std::shared_ptr<Block>& block) {
    return registry().buildBlock(type, block);
}

}  // namespace Jetstream
