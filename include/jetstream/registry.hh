#ifndef JETSTREAM_REGISTRY_HH
#define JETSTREAM_REGISTRY_HH

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "jetstream/provider.hh"
#include "jetstream/runtime.hh"
#include "jetstream/parser.hh"
#include "jetstream/module.hh"
#include "jetstream/block.hh"
#include "jetstream/flowgraph.hh"
#include "jetstream/benchmark.hh"

namespace Jetstream {

class JETSTREAM_API Registry {
 public:
    using ModuleFactory = std::function<std::shared_ptr<Module>(const std::shared_ptr<Flowgraph::Environment>&,
                                                                const std::shared_ptr<Flowgraph::View>&)>;
    using BlockFactory = std::function<std::shared_ptr<Block>()>;
    using BenchmarkFactory = std::function<std::vector<Benchmark::Case>()>;

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
        std::string summary = "No summary.";
        std::string description = "No description.";
        std::string content;

        JST_SERDES(title, summary, description);
    };

    struct BenchmarkRegistration {
        std::string moduleType;
        const void* owner = nullptr;
        BenchmarkFactory factory;
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
    static Result RegisterBenchmark(const std::string& moduleType,
                                    BenchmarkFactory factory,
                                    const void* owner);

    static Result UnregisterModule(const std::string& type,
                                    DeviceType device,
                                    RuntimeType runtime,
                                    const ProviderType& provider);
    static Result UnregisterBlock(const std::string& type);
    static Result UnregisterFlowgraph(const std::string& key);
    static Result UnregisterBenchmark(const std::string& moduleType,
                                      const void* owner);

    static std::vector<ModuleRegistration>
        ListAvailableModules(const std::string& type = "",
                             std::optional<DeviceType> device = std::nullopt,
                             std::optional<RuntimeType> runtime = std::nullopt,
                             const ProviderType& provider = "");
    static std::vector<BlockRegistration>
        ListAvailableBlocks(const std::string& type = "");
    static std::vector<FlowgraphRegistration>
        ListAvailableFlowgraphs(const std::string& key = "");
    static std::vector<BenchmarkRegistration>
        ListAvailableBenchmarks(const std::string& moduleType = "");

    static Result BuildModule(const std::string& type,
                               const DeviceType& device,
                               const RuntimeType& runtime,
                               const ProviderType& provider,
                               std::shared_ptr<Module>& module,
                               const std::shared_ptr<Flowgraph::Environment>& environment = nullptr,
                               const std::shared_ptr<Flowgraph::View>& view = nullptr);
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
                [device, runtime, provider](const std::shared_ptr<::Jetstream::Flowgraph::Environment>& environment, \
                                            const std::shared_ptr<::Jetstream::Flowgraph::View>& view) { \
                    const auto impl = std::make_shared<impl_type>(); \
                    const auto runtimeContext = std::static_pointer_cast<::Jetstream::Runtime::Context>(impl); \
                    const auto schedulerContext = std::static_pointer_cast<::Jetstream::Scheduler::Context>(impl); \
                    const auto context = std::make_shared<::Jetstream::Module::Context>(runtimeContext, \
                                                                                        schedulerContext, \
                                                                                        environment, \
                                                                                        view); \
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

#define JST_DETAIL_REGISTER_EXAMPLE(key_val, content_val, id) \
    namespace { \
    [[maybe_unused]] const ::Jetstream::Result JST_DETAIL_CONCAT(__jst_register_example_, id) = \
        ::Jetstream::Registry::QueueStaticRegistration([]() { \
            ::Jetstream::Registry::FlowgraphRegistration record; \
            record.key = key_val; \
            record.title = record.key; \
            record.content = content_val; \
            ::Jetstream::Parser::Map data; \
            auto result = ::Jetstream::Parser::YamlDecode(record.content, data); \
            if (result != ::Jetstream::Result::SUCCESS) { \
                return result; \
            } \
            result = record.deserialize(data); \
            if (result != ::Jetstream::Result::SUCCESS) { \
                return result; \
            } \
            if (record.title.empty()) { \
                record.title = record.key; \
            } \
            return ::Jetstream::Registry::RegisterFlowgraph(record.key, record); \
        }); \
    }

#define JST_REGISTER_EXAMPLE(key_val, content_val) \
    JST_DETAIL_REGISTER_EXAMPLE(key_val, content_val, __COUNTER__)

#define JST_DETAIL_REGISTER_BENCHMARKS(module_type_val, id) \
    static std::vector<::Jetstream::Benchmark::Case> \
    JST_DETAIL_CONCAT(__jst_benchmark_specs_, id)(); \
    namespace { \
    [[maybe_unused]] const ::Jetstream::Result JST_DETAIL_CONCAT(__jst_register_benchmarks_, id) = \
        ::Jetstream::Registry::QueueStaticRegistration([]() { \
            static const int JST_DETAIL_CONCAT(__jst_benchmark_owner_, id) = 0; \
            const std::string moduleType = module_type_val; \
            return ::Jetstream::Registry::RegisterBenchmark( \
                moduleType, \
                &JST_DETAIL_CONCAT(__jst_benchmark_specs_, id), \
                &JST_DETAIL_CONCAT(__jst_benchmark_owner_, id) \
            ); \
        }); \
    } \
    static std::vector<::Jetstream::Benchmark::Case> \
    JST_DETAIL_CONCAT(__jst_benchmark_specs_, id)()

#define JST_BENCHMARKS(module_type_val) \
    JST_DETAIL_REGISTER_BENCHMARKS(module_type_val, __COUNTER__)

#endif  // JETSTREAM_REGISTRY_HH
