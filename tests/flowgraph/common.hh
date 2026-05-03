#ifndef JETSTREAM_TESTS_FLOWGRAPH_COMMON_HH
#define JETSTREAM_TESTS_FLOWGRAPH_COMMON_HH

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "flowgraph_fixture.hh"
#include "jetstream/detail/module_impl.hh"
#include "jetstream/domains/core/add/block.hh"
#include "jetstream/domains/dsp/signal_generator/block.hh"
#include "jetstream/domains/dsp/signal_generator/module.hh"
#include "jetstream/flowgraph.hh"
#include "jetstream/module_context.hh"
#include "jetstream/registry.hh"
#include "jetstream/runtime_context_native_cpu.hh"
#include "jetstream/scheduler_context.hh"

namespace TestFlowgraph {

using namespace Jetstream;

struct SimpleMetaFixture {
    U64 order = 0;
    std::string label;

    JST_SERDES(order, label);
};

inline constexpr auto kSignalGeneratorTestProvider = "test-alt";

struct SignalGeneratorTestProviderImpl : Module::Impl,
                                         DynamicConfig<Modules::SignalGenerator>,
                                         NativeCpuRuntimeContext,
                                         Scheduler::Context {
    Result validate() override {
        if (signalDataType != "F32" && signalDataType != "CF32") {
            return Result::ERROR;
        }

        if (bufferSize == 0) {
            return Result::ERROR;
        }

        return Result::SUCCESS;
    }

    Result define() override {
        return defineInterfaceOutput("signal");
    }

    Result create() override {
        JST_CHECK(signal.create(device(), NameToDataType(signalDataType), {bufferSize}));
        outputs()["signal"].produced(name(), "signal", signal);

        return Result::SUCCESS;
    }

    Result destroy() override {
        return Result::SUCCESS;
    }

    Result reconfigure() override {
        return Result::RECREATE;
    }

    Tensor signal;
};

inline Result RegisterSignalGeneratorTestProvider() {
    static const Result result = Registry::RegisterModule(
        "signal_generator",
        DeviceType::CPU,
        RuntimeType::NATIVE,
        kSignalGeneratorTestProvider,
        []() -> std::shared_ptr<Module> {
            const auto impl = std::make_shared<SignalGeneratorTestProviderImpl>();
            const auto runtimeContext = std::static_pointer_cast<Runtime::Context>(impl);
            const auto schedulerContext = std::static_pointer_cast<Scheduler::Context>(impl);
            const auto context = std::make_shared<Module::Context>(runtimeContext, schedulerContext);
            const auto stagedConfig = std::static_pointer_cast<Module::Config>(impl);
            const auto candidateConfig = std::static_pointer_cast<Module::Config>(impl->candidate());

            return std::make_shared<Module>(DeviceType::CPU,
                                            RuntimeType::NATIVE,
                                            kSignalGeneratorTestProvider,
                                            impl,
                                            context,
                                            stagedConfig,
                                            candidateConfig);
        });

    return result;
}

}  // namespace TestFlowgraph

#endif  // JETSTREAM_TESTS_FLOWGRAPH_COMMON_HH
