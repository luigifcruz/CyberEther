#ifndef JETSTREAM_TESTS_SUPPORT_SYNTHETIC_GRAPH_HH
#define JETSTREAM_TESTS_SUPPORT_SYNTHETIC_GRAPH_HH

#include <cstddef>
#include <functional>
#include <memory>
#include <string>

#include "jetstream/detail/block_impl.hh"
#include "jetstream/detail/module_impl.hh"
#include "jetstream/module_context.hh"
#include "jetstream/registry.hh"
#include "jetstream/runtime_context_native_cpu.hh"
#include "jetstream/scheduler_context.hh"

namespace TestFlowgraph {

using namespace Jetstream;

inline constexpr auto kSyntheticSourceType = "flowgraph_test_source";
inline constexpr auto kSyntheticPassType = "flowgraph_test_pass";
inline constexpr auto kSyntheticMergeType = "flowgraph_test_merge";
inline constexpr auto kSyntheticIsolatedType = "flowgraph_test_isolated";
inline constexpr auto kSyntheticFaultType = "flowgraph_test_fault";
inline constexpr auto kSyntheticSourceTestProvider = "flowgraph-test-alt";

enum class SyntheticFaultPoint {
    None,
    BlockConfigure,
    BlockDefine,
    BlockCreate,
    BlockDestroy,
    ModuleDefine,
    ModuleCreate,
    ModuleDestroy,
    ModuleReconfigure,
    ModulePresent,
};

struct SyntheticFaultState {
    SyntheticFaultPoint next = SyntheticFaultPoint::None;
    U64 blockConfigureCalls = 0;
    U64 blockDefineCalls = 0;
    U64 blockCreateCalls = 0;
    U64 blockDestroyCalls = 0;
    U64 moduleDefineCalls = 0;
    U64 moduleCreateCalls = 0;
    U64 moduleDestroyCalls = 0;
    U64 moduleReconfigureCalls = 0;
    U64 modulePresentCalls = 0;
    std::function<void()> onBlockCreate;

    void reset() {
        *this = {};
    }

    void failNext(const SyntheticFaultPoint point) {
        next = point;
    }

    bool consume(const SyntheticFaultPoint point) {
        if (next != point) {
            return false;
        }

        next = SyntheticFaultPoint::None;
        return true;
    }
};

inline SyntheticFaultState& syntheticFaultState() {
    static SyntheticFaultState state;
    return state;
}

struct SimpleMetaFixture {
    U64 order = 0;
    std::string label;

    JST_SERDES(order, label);
};

struct SyntheticSourceBlockConfig : Block::Config {
    U64 bufferSize = 8192;
    F32 value = 1.0f;

    JST_BLOCK_TYPE(flowgraph_test_source)
    JST_BLOCK_DOMAIN("Test")
    JST_BLOCK_PARAMS(bufferSize, value)
    JST_BLOCK_DESCRIPTION("Synthetic Source",
                          "Produces a test-owned tensor.",
                          "Flowgraph fixture source block.")
};

struct SyntheticPassBlockConfig : Block::Config {
    JST_BLOCK_TYPE(flowgraph_test_pass)
    JST_BLOCK_DOMAIN("Test")
    JST_BLOCK_DESCRIPTION("Synthetic Pass",
                          "Passes a test-owned tensor through.",
                          "Flowgraph fixture pass-through block.")

    Result serialize(Parser::Map&) const override { return Result::SUCCESS; }
    Result deserialize(const Parser::Map&) override { return Result::SUCCESS; }
    std::size_t hash() const override { return 0; }
};

struct SyntheticMergeBlockConfig : Block::Config {
    JST_BLOCK_TYPE(flowgraph_test_merge)
    JST_BLOCK_DOMAIN("Test")
    JST_BLOCK_DESCRIPTION("Synthetic Merge",
                          "Merges two test-owned graph branches.",
                          "Flowgraph fixture merge block.")

    Result serialize(Parser::Map&) const override { return Result::SUCCESS; }
    Result deserialize(const Parser::Map&) override { return Result::SUCCESS; }
    std::size_t hash() const override { return 0; }
};

struct SyntheticIsolatedBlockConfig : Block::Config {
    JST_BLOCK_TYPE(flowgraph_test_isolated)
    JST_BLOCK_DOMAIN("Test")
    JST_BLOCK_DESCRIPTION("Synthetic Isolated",
                          "Represents an isolated graph node.",
                          "Flowgraph fixture block without ports or modules.")

    Result serialize(Parser::Map&) const override { return Result::SUCCESS; }
    Result deserialize(const Parser::Map&) override { return Result::SUCCESS; }
    std::size_t hash() const override { return 0; }
};

struct SyntheticFaultBlockConfig : Block::Config {
    U64 revision = 0;

    JST_BLOCK_TYPE(flowgraph_test_fault)
    JST_BLOCK_DOMAIN("Test")
    JST_BLOCK_PARAMS(revision)
    JST_BLOCK_DESCRIPTION("Synthetic Fault Source",
                          "Injects test-owned lifecycle failures.",
                          "Flowgraph fixture fault source block.")
};

namespace Detail {

struct SyntheticSourceModuleConfig : Module::Config {
    U64 bufferSize = 8192;
    F32 value = 1.0f;

    JST_MODULE_TYPE(flowgraph_test_source)
    JST_MODULE_PARAMS(bufferSize, value)
};

struct SyntheticPassModuleConfig : Module::Config {
    JST_MODULE_TYPE(flowgraph_test_pass)

    Result serialize(Parser::Map&) const override { return Result::SUCCESS; }
    Result deserialize(const Parser::Map&) override { return Result::SUCCESS; }
    std::size_t hash() const override { return 0; }
};

struct SyntheticMergeModuleConfig : Module::Config {
    JST_MODULE_TYPE(flowgraph_test_merge)

    Result serialize(Parser::Map&) const override { return Result::SUCCESS; }
    Result deserialize(const Parser::Map&) override { return Result::SUCCESS; }
    std::size_t hash() const override { return 0; }
};

struct SyntheticFaultModuleConfig : Module::Config {
    U64 revision = 0;

    JST_MODULE_TYPE(flowgraph_test_fault)
    JST_MODULE_PARAMS(revision)
};

struct SyntheticSourceModule : Module::Impl,
                               DynamicConfig<SyntheticSourceModuleConfig>,
                               NativeCpuRuntimeContext,
                               Scheduler::Context {
    Result validate() override {
        return candidate()->bufferSize == 0 ? Result::ERROR : Result::SUCCESS;
    }

    Result define() override {
        return defineInterfaceOutput("signal");
    }

    Result create() override {
        JST_CHECK(signal.create(device(), DataType::F32, {bufferSize}));
        signal.at<F32>(0) = value;
        outputs()["signal"].produced(name(), "signal", signal);
        return Result::SUCCESS;
    }

    Result computeSubmit() override {
        return Result::SUCCESS;
    }

    Tensor signal;
};

struct SyntheticPassModule : Module::Impl,
                             DynamicConfig<SyntheticPassModuleConfig>,
                             NativeCpuRuntimeContext,
                             Scheduler::Context {
    Result define() override {
        JST_CHECK(defineInterfaceInput("buffer"));
        return defineInterfaceOutput("buffer");
    }

    Result create() override {
        output = inputs().at("buffer").tensor.clone();
        outputs()["buffer"].produced(name(), "buffer", output);
        return Result::SUCCESS;
    }

    Result computeSubmit() override {
        return Result::SUCCESS;
    }

    Tensor output;
};

struct SyntheticMergeModule : Module::Impl,
                              DynamicConfig<SyntheticMergeModuleConfig>,
                              NativeCpuRuntimeContext,
                              Scheduler::Context {
    Result define() override {
        JST_CHECK(defineInterfaceInput("a"));
        JST_CHECK(defineInterfaceInput("b"));
        return defineInterfaceOutput("sum");
    }

    Result create() override {
        if (inputs().at("a").tensor.shape() != inputs().at("b").tensor.shape()) {
            return Result::ERROR;
        }

        output = inputs().at("a").tensor.clone();
        outputs()["sum"].produced(name(), "sum", output);
        return Result::SUCCESS;
    }

    Result computeSubmit() override {
        return Result::SUCCESS;
    }

    Tensor output;
};

struct SyntheticFaultModule : Module::Impl,
                              DynamicConfig<SyntheticFaultModuleConfig>,
                              NativeCpuRuntimeContext,
                              Scheduler::Context {
    Result define() override {
        auto& state = syntheticFaultState();
        state.moduleDefineCalls += 1;
        if (state.consume(SyntheticFaultPoint::ModuleDefine)) {
            JST_ERROR("[FLOWGRAPH_TEST_FAULT] Forced module define failure.");
            return Result::ERROR;
        }

        return defineInterfaceOutput("out");
    }

    Result create() override {
        auto& state = syntheticFaultState();
        state.moduleCreateCalls += 1;
        if (state.consume(SyntheticFaultPoint::ModuleCreate)) {
            JST_ERROR("[FLOWGRAPH_TEST_FAULT] Forced module create failure.");
            return Result::ERROR;
        }

        JST_CHECK(output.create(DeviceType::CPU, DataType::F32, {1}));
        output.at<F32>(0) = 1.0f;
        outputs()["out"].produced(name(), "out", output);
        return Result::SUCCESS;
    }

    Result destroy() override {
        auto& state = syntheticFaultState();
        state.moduleDestroyCalls += 1;
        if (state.consume(SyntheticFaultPoint::ModuleDestroy)) {
            JST_ERROR("[FLOWGRAPH_TEST_FAULT] Forced module destroy failure.");
            return Result::ERROR;
        }

        return Result::SUCCESS;
    }

    Result reconfigure() override {
        auto& state = syntheticFaultState();
        state.moduleReconfigureCalls += 1;
        if (state.consume(SyntheticFaultPoint::ModuleReconfigure)) {
            JST_ERROR("[FLOWGRAPH_TEST_FAULT] Forced module reconfigure failure.");
            return Result::ERROR;
        }

        return Result::SUCCESS;
    }

    Result presentSubmit() override {
        auto& state = syntheticFaultState();
        state.modulePresentCalls += 1;
        if (state.consume(SyntheticFaultPoint::ModulePresent)) {
            JST_ERROR("[FLOWGRAPH_TEST_FAULT] Forced module present failure.");
            return Result::ERROR;
        }

        return Result::SUCCESS;
    }

    Result computeSubmit() override {
        return Result::SUCCESS;
    }

    Tensor output;
};

struct SyntheticSourceBlock : Block::Impl,
                              DynamicConfig<SyntheticSourceBlockConfig> {
    Result configure() override {
        moduleConfig->bufferSize = bufferSize;
        moduleConfig->value = value;
        return Result::SUCCESS;
    }

    Result define() override {
        JST_CHECK(defineInterfaceOutput(
            "signal", "Output", "Synthetic output tensor."));
        JST_CHECK(defineInterfaceConfig(
            "bufferSize", "Buffer Size", "Output tensor size.", "int"));
        return defineInterfaceConfig("value", "Value", "Output tensor value.", "float");
    }

    Result create() override {
        JST_CHECK(moduleCreate("source", moduleConfig, {}));
        return moduleExposeOutput("signal", {"source", "signal"});
    }

    std::shared_ptr<SyntheticSourceModuleConfig> moduleConfig =
        std::make_shared<SyntheticSourceModuleConfig>();
};

struct SyntheticPassBlock : Block::Impl,
                            DynamicConfig<SyntheticPassBlockConfig> {
    Result define() override {
        JST_CHECK(defineInterfaceInput("buffer", "Input", "Synthetic input tensor."));
        return defineInterfaceOutput("buffer", "Output", "Synthetic output tensor.");
    }

    Result create() override {
        const auto config = std::make_shared<SyntheticPassModuleConfig>();
        JST_CHECK(moduleCreate("pass", config, {{"buffer", inputs().at("buffer")}}));
        return moduleExposeOutput("buffer", {"pass", "buffer"});
    }
};

struct SyntheticMergeBlock : Block::Impl,
                             DynamicConfig<SyntheticMergeBlockConfig> {
    Result define() override {
        JST_CHECK(defineInterfaceInput(
            "a", "Input A", "First synthetic input tensor."));
        JST_CHECK(defineInterfaceInput(
            "b", "Input B", "Second synthetic input tensor."));
        return defineInterfaceOutput("sum", "Output", "Synthetic merged tensor.");
    }

    Result create() override {
        const auto config = std::make_shared<SyntheticMergeModuleConfig>();
        JST_CHECK(moduleCreate("merge", config, {
            {"a", inputs().at("a")},
            {"b", inputs().at("b")},
        }));
        return moduleExposeOutput("sum", {"merge", "sum"});
    }
};

struct SyntheticIsolatedBlock : Block::Impl,
                                 DynamicConfig<SyntheticIsolatedBlockConfig> {};

struct SyntheticFaultBlock : Block::Impl,
                             DynamicConfig<SyntheticFaultBlockConfig> {
    Result configure() override {
        auto& state = syntheticFaultState();
        state.blockConfigureCalls += 1;
        if (state.consume(SyntheticFaultPoint::BlockConfigure)) {
            JST_ERROR("[FLOWGRAPH_TEST_FAULT] Forced block configure failure.");
            return Result::ERROR;
        }

        moduleConfig->revision = revision;
        return Result::SUCCESS;
    }

    Result define() override {
        auto& state = syntheticFaultState();
        state.blockDefineCalls += 1;
        if (state.consume(SyntheticFaultPoint::BlockDefine)) {
            JST_ERROR("[FLOWGRAPH_TEST_FAULT] Forced block define failure.");
            return Result::ERROR;
        }

        JST_CHECK(defineInterfaceOutput(
            "out", "Output", "Synthetic fault output tensor."));
        return defineInterfaceConfig(
            "revision", "Revision", "Synthetic configuration revision.", "int");
    }

    Result create() override {
        auto& state = syntheticFaultState();
        state.blockCreateCalls += 1;
        if (state.onBlockCreate) {
            state.onBlockCreate();
        }
        if (state.consume(SyntheticFaultPoint::BlockCreate)) {
            JST_ERROR("[FLOWGRAPH_TEST_FAULT] Forced block create failure.");
            return Result::ERROR;
        }

        JST_CHECK(moduleCreate("fault", moduleConfig, {}));
        return moduleExposeOutput("out", {"fault", "out"});
    }

    Result destroy() override {
        auto& state = syntheticFaultState();
        state.blockDestroyCalls += 1;
        if (state.consume(SyntheticFaultPoint::BlockDestroy)) {
            JST_ERROR("[FLOWGRAPH_TEST_FAULT] Forced block destroy failure.");
            return Result::ERROR;
        }

        return Result::SUCCESS;
    }

    // TODO: Cover BlockDestroy once Block::destroy() invokes Block::Impl::destroy().
    std::shared_ptr<SyntheticFaultModuleConfig> moduleConfig =
        std::make_shared<SyntheticFaultModuleConfig>();
};

template<typename Impl>
std::shared_ptr<Module> BuildModule(
    const DeviceType device,
    const RuntimeType runtime,
    const ProviderType& provider,
    const std::shared_ptr<Flowgraph::Environment>& environment,
    const std::shared_ptr<Flowgraph::View>& view) {
    const auto impl = std::make_shared<Impl>();
    const auto runtimeContext = std::static_pointer_cast<Runtime::Context>(impl);
    const auto schedulerContext = std::static_pointer_cast<Scheduler::Context>(impl);
    const auto context = std::make_shared<Module::Context>(runtimeContext,
                                                           schedulerContext,
                                                           environment,
                                                           view);
    const auto stagedConfig = std::static_pointer_cast<Module::Config>(impl);
    const auto candidateConfig =
        std::static_pointer_cast<Module::Config>(impl->candidate());
    return std::make_shared<Module>(device,
                                    runtime,
                                    provider,
                                    impl,
                                    context,
                                    stagedConfig,
                                    candidateConfig);
}

template<typename Impl>
Result RegisterModule(const ProviderType& provider) {
    const auto sample = std::make_shared<Impl>();
    return Registry::RegisterModule(
        sample->type(),
        DeviceType::CPU,
        RuntimeType::NATIVE,
        provider,
        [provider](const std::shared_ptr<Flowgraph::Environment>& environment,
                   const std::shared_ptr<Flowgraph::View>& view) {
            return BuildModule<Impl>(DeviceType::CPU,
                                     RuntimeType::NATIVE,
                                     provider,
                                     environment,
                                     view);
        });
}

template<typename Impl>
Result RegisterBlock() {
    const auto sample = std::make_shared<Impl>();
    return Registry::RegisterBlock(
        sample->type(),
        sample->domain(),
        sample->title(),
        sample->summary(),
        sample->description(),
        []() {
            const auto impl = std::make_shared<Impl>();
            const auto stagedConfig = std::static_pointer_cast<Block::Config>(impl);
            const auto candidateConfig =
                std::static_pointer_cast<Block::Config>(impl->candidate());
            return std::make_shared<Block>(impl, stagedConfig, candidateConfig);
        });
}

}  // namespace Detail

inline Result RegisterSyntheticGraph() {
    static const Result result = []() {
        JST_CHECK(Detail::RegisterModule<Detail::SyntheticSourceModule>("generic"));
        JST_CHECK(Detail::RegisterModule<Detail::SyntheticSourceModule>(
            kSyntheticSourceTestProvider));
        JST_CHECK(Detail::RegisterModule<Detail::SyntheticPassModule>("generic"));
        JST_CHECK(Detail::RegisterModule<Detail::SyntheticMergeModule>("generic"));
        JST_CHECK(Detail::RegisterModule<Detail::SyntheticFaultModule>("generic"));
        JST_CHECK(Detail::RegisterBlock<Detail::SyntheticSourceBlock>());
        JST_CHECK(Detail::RegisterBlock<Detail::SyntheticPassBlock>());
        JST_CHECK(Detail::RegisterBlock<Detail::SyntheticMergeBlock>());
        JST_CHECK(Detail::RegisterBlock<Detail::SyntheticIsolatedBlock>());
        JST_CHECK(Detail::RegisterBlock<Detail::SyntheticFaultBlock>());
        return Result::SUCCESS;
    }();

    return result;
}

}  // namespace TestFlowgraph

#endif  // JETSTREAM_TESTS_SUPPORT_SYNTHETIC_GRAPH_HH
