#ifndef JETSTREAM_REGISTRY_HH
#define JETSTREAM_REGISTRY_HH

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <unordered_map>

#include "jetstream/provider.hh"
#include "jetstream/runtime.hh"
#include "jetstream/module.hh"
#include "jetstream/block.hh"

namespace Jetstream {

class JETSTREAM_API Registry {
 public:
    using ModuleFactory = std::function<std::shared_ptr<Module>()>;
    using BlockFactory = std::function<std::shared_ptr<Block>()>;

    struct ModuleRegistration {
        std::string type;
        DeviceType device;
        RuntimeType runtime;
        ProviderType provider;
        ModuleFactory factory;
    };

    struct BlockRegistration {
        std::string type;
        std::string title;
        std::string summary;
        std::string description;
        BlockFactory factory;
    };

    struct FlowgraphRegistration {
        std::string key;
        std::string title;
        std::string summary;
        std::string description;
        std::string content;
    };

    static Result RegisterModule(const std::string& type,
                                  DeviceType device,
                                  RuntimeType runtime,
                                  const ProviderType& provider,
                                  ModuleFactory factory);
    static Result RegisterBlock(const std::string& type,
                                const std::string& title,
                                const std::string& summary,
                                const std::string& description,
                                BlockFactory factory);
    static Result RegisterFlowgraph(const std::string& key,
                                    const FlowgraphRegistration& metadata);

    static std::vector<ModuleRegistration>
        ListAvailableModules(const std::string& type = "",
                             std::optional<DeviceType> device = std::nullopt,
                             std::optional<RuntimeType> runtime = std::nullopt,
                             const ProviderType& provider = "");
    static std::vector<BlockRegistration>
        ListAvailableBlocks(const std::string& type = "");
    static std::vector<FlowgraphRegistration>
        ListAvailableFlowgraphs(const std::string& key = "");

    static Result BuildModule(const std::string& type,
                               const DeviceType& device,
                               const RuntimeType& runtime,
                               const ProviderType& provider,
                               std::shared_ptr<Module>& module);
    static Result BuildBlock(const std::string& type, std::shared_ptr<Block>& block);

 private:
    struct Impl;
    static Impl& registry();
};

}  // namespace Jetstream

#define JST_DETAIL_CONCAT_IMPL(x, y) x##y
#define JST_DETAIL_CONCAT(x, y) JST_DETAIL_CONCAT_IMPL(x, y)

#define JST_DETAIL_REGISTER_MODULE(impl_type, device_val, runtime_val, provider_val, id) \
    namespace { \
    static void JST_DETAIL_CONCAT(__jst_register_module_, id)() __attribute__((constructor)) __attribute__((used)); \
    static void JST_DETAIL_CONCAT(__jst_register_module_, id)() { \
        const auto& module = std::make_shared<impl_type>(); \
        if (::Jetstream::Registry::RegisterModule( \
            module->type(), device_val, runtime_val, provider_val, \
            []() { \
                const auto& impl = std::make_shared<impl_type>(); \
                const auto& runtimeContext = std::static_pointer_cast<Runtime::Context>(impl); \
                const auto& schedulerContext = std::static_pointer_cast<Scheduler::Context>(impl); \
                const auto& context = std::make_shared<Module::Context>(runtimeContext, schedulerContext); \
                const auto& stagedConfig = std::static_pointer_cast<Module::Config>(impl); \
                const auto& candidateConfig = std::static_pointer_cast<Module::Config>(impl->candidate()); \
                return std::make_shared<Module>(device_val, runtime_val, provider_val, impl, context, stagedConfig, candidateConfig); \
            } \
        ) != ::Jetstream::Result::SUCCESS) { \
            std::abort(); \
        } \
    } \
    }

#define JST_REGISTER_MODULE(impl_type, device_val, runtime_val, provider_val) \
    JST_DETAIL_REGISTER_MODULE(impl_type, device_val, runtime_val, provider_val, __COUNTER__)

#define JST_DETAIL_REGISTER_BLOCK(impl_type, id) \
    namespace { \
    static void JST_DETAIL_CONCAT(__jst_register_block_, id)() __attribute__((constructor)) __attribute__((used)); \
    static void JST_DETAIL_CONCAT(__jst_register_block_, id)() { \
        const auto& block = std::make_shared<impl_type>(); \
        if (::Jetstream::Registry::RegisterBlock( \
            block->type(), block->title(), block->summary(), block->description(), \
            []() { \
                const auto& impl = std::make_shared<impl_type>(); \
                const auto& stagedConfig = std::static_pointer_cast<Block::Config>(impl); \
                const auto& candidateConfig = std::static_pointer_cast<Block::Config>(impl->candidate()); \
                return std::make_shared<Block>(impl, stagedConfig, candidateConfig); \
            } \
        ) != ::Jetstream::Result::SUCCESS) { \
            std::abort(); \
        } \
    } \
    }

#define JST_REGISTER_BLOCK(impl_type) \
    JST_DETAIL_REGISTER_BLOCK(impl_type, __COUNTER__)

#endif  // JETSTREAM_REGISTRY_HH
