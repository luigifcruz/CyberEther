#ifndef JETSTREAM_REGISTRY_HH
#define JETSTREAM_REGISTRY_HH

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

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
        std::string domain;
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

    static Result QueueStaticRegistration(std::function<Result()> callback);
    static Result DrainStaticRegistrations();
    static Result DiscardStaticRegistrations();

    static Result RegisterModule(const std::string& type,
                                  DeviceType device,
                                  RuntimeType runtime,
                                  const ProviderType& provider,
                                  ModuleFactory factory);
    static Result RegisterBlock(const std::string& type,
                                const std::string& domain,
                                const std::string& title,
                                const std::string& summary,
                                const std::string& description,
                                BlockFactory factory);
    static Result RegisterFlowgraph(const std::string& key,
                                    const FlowgraphRegistration& metadata);

    static Result UnregisterModule(const std::string& type,
                                    DeviceType device,
                                    RuntimeType runtime,
                                    const ProviderType& provider);
    static Result UnregisterBlock(const std::string& type);
    static Result UnregisterFlowgraph(const std::string& key);

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
    [[maybe_unused]] const ::Jetstream::Result JST_DETAIL_CONCAT(__jst_register_module_, id) = \
        ::Jetstream::Registry::QueueStaticRegistration([]() { \
            const ::Jetstream::DeviceType device = device_val; \
            const ::Jetstream::RuntimeType runtime = runtime_val; \
            const ::Jetstream::ProviderType provider = provider_val; \
            const auto module = std::make_shared<impl_type>(); \
            return ::Jetstream::Registry::RegisterModule( \
                module->type(), \
                device, \
                runtime, \
                provider, \
                [device, runtime, provider]() { \
                    const auto impl = std::make_shared<impl_type>(); \
                    const auto runtimeContext = std::static_pointer_cast<::Jetstream::Runtime::Context>(impl); \
                    const auto schedulerContext = std::static_pointer_cast<::Jetstream::Scheduler::Context>(impl); \
                    const auto context = std::make_shared<::Jetstream::Module::Context>(runtimeContext, schedulerContext); \
                    const auto stagedConfig = std::static_pointer_cast<::Jetstream::Module::Config>(impl); \
                    const auto candidateConfig = std::static_pointer_cast<::Jetstream::Module::Config>(impl->candidate()); \
                    return std::make_shared<::Jetstream::Module>(device, \
                                                                 runtime, \
                                                                 provider, \
                                                                 impl, \
                                                                 context, \
                                                                 stagedConfig, \
                                                                 candidateConfig); \
                } \
            ); \
        }); \
    }

#define JST_REGISTER_MODULE(impl_type, device_val, runtime_val, provider_val) \
    JST_DETAIL_REGISTER_MODULE(impl_type, device_val, runtime_val, provider_val, __COUNTER__)

#define JST_DETAIL_REGISTER_BLOCK(impl_type, id) \
    namespace { \
    [[maybe_unused]] const ::Jetstream::Result JST_DETAIL_CONCAT(__jst_register_block_, id) = \
        ::Jetstream::Registry::QueueStaticRegistration([]() { \
            const auto block = std::make_shared<impl_type>(); \
            return ::Jetstream::Registry::RegisterBlock( \
                block->type(), \
                block->domain(), \
                block->title(), \
                block->summary(), \
                block->description(), \
                []() { \
                    const auto impl = std::make_shared<impl_type>(); \
                    const auto stagedConfig = std::static_pointer_cast<::Jetstream::Block::Config>(impl); \
                    const auto candidateConfig = std::static_pointer_cast<::Jetstream::Block::Config>(impl->candidate()); \
                    return std::make_shared<::Jetstream::Block>(impl, stagedConfig, candidateConfig); \
                } \
            ); \
        }); \
    }

#define JST_REGISTER_BLOCK(impl_type) \
    JST_DETAIL_REGISTER_BLOCK(impl_type, __COUNTER__)

#define JST_DETAIL_REGISTER_EXAMPLE(key_val, title_val, summary_val, description_val, content_val, id) \
    namespace { \
    [[maybe_unused]] const ::Jetstream::Result JST_DETAIL_CONCAT(__jst_register_example_, id) = \
        ::Jetstream::Registry::QueueStaticRegistration([]() { \
            const std::string key = key_val; \
            const std::string title = title_val; \
            const std::string summary = summary_val; \
            const std::string description = description_val; \
            const std::string content = content_val; \
            return ::Jetstream::Registry::RegisterFlowgraph( \
                key, \
                {key, title, summary, description, content} \
            ); \
        }); \
    }

#define JST_REGISTER_EXAMPLE(key_val, title_val, summary_val, description_val, content_val) \
    JST_DETAIL_REGISTER_EXAMPLE(key_val, title_val, summary_val, description_val, content_val, __COUNTER__)

#endif  // JETSTREAM_REGISTRY_HH
