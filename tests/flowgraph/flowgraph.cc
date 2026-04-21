#include <catch2/catch_approx.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include "flowgraph_fixture.hh"
#include "jetstream/detail/module_impl.hh"
#include "jetstream/flowgraph.hh"
#include "jetstream/logger.hh"
#include "jetstream/module_context.hh"
#include "jetstream/registry.hh"
#include "jetstream/runtime_context_native_cpu.hh"
#include "jetstream/scheduler_context.hh"

#include "jetstream/domains/dsp/signal_generator/block.hh"
#include "jetstream/domains/dsp/signal_generator/module.hh"
#include "jetstream/domains/core/add/block.hh"

using namespace Jetstream;

namespace {

struct StackDockFlowgraphFixture {
    U64 order = 0;

    JST_SERDES(order);
};

struct StackDockSurfaceFixture {
    std::string block;
    std::string surface;
    U64 order = 0;

    JST_SERDES(block, surface, order);
};

struct StackDockLayoutFixture {
    std::string direction;
    F32 ratio = 0.5f;
    std::vector<StackDockFlowgraphFixture> flowgraphs;
    std::vector<StackDockSurfaceFixture> surfaces;
    std::unordered_map<std::string, StackDockLayoutFixture> children;

    JST_SERDES(direction, ratio, flowgraphs, surfaces, children);
};

struct StackMetaFixture {
    std::string title;
    F32 x = 0.0f;
    F32 y = 0.0f;
    F32 width = 0.0f;
    F32 height = 0.0f;
    std::unordered_map<std::string, StackDockLayoutFixture> layout;

    JST_SERDES(title, x, y, width, height, layout);
};

constexpr auto kSignalGeneratorTestProvider = "test-alt";

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

Result RegisterSignalGeneratorTestProvider() {
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

}  // namespace

TEST_CASE_METHOD(FlowgraphFixture, "Block creation and destruction", "[flowgraph]") {
    SECTION("create single block") {
        auto result = flowgraph->blockCreate("gen1", "signal_generator", {}, {});
        REQUIRE(result == Result::SUCCESS);
        REQUIRE(flowgraph->blockList().size() == 1);
        REQUIRE(flowgraph->blockList().contains("gen1"));
        REQUIRE(flowgraph->blockList().at("gen1")->state() == Block::State::Created);
    }

    SECTION("create multiple blocks") {
        REQUIRE(flowgraph->blockCreate("gen1", "signal_generator", {}, {}) == Result::SUCCESS);
        REQUIRE(flowgraph->blockCreate("gen2", "signal_generator", {}, {}) == Result::SUCCESS);
        REQUIRE(flowgraph->blockList().size() == 2);
        REQUIRE(flowgraph->blockList().at("gen1")->state() == Block::State::Created);
        REQUIRE(flowgraph->blockList().at("gen2")->state() == Block::State::Created);
    }

    SECTION("destroy block") {
        REQUIRE(flowgraph->blockCreate("gen1", "signal_generator", {}, {}) == Result::SUCCESS);
        REQUIRE(flowgraph->blockDestroy("gen1") == Result::SUCCESS);
        REQUIRE(flowgraph->blockList().empty());
    }

    SECTION("create duplicate block fails") {
        REQUIRE(flowgraph->blockCreate("gen1", "signal_generator", {}, {}) == Result::SUCCESS);
        REQUIRE(flowgraph->blockCreate("gen1", "signal_generator", {}, {}) == Result::ERROR);
    }

    SECTION("destroy nonexistent block fails") {
        REQUIRE(flowgraph->blockDestroy("nonexistent") == Result::ERROR);
    }

    SECTION("create invalid block type fails") {
        auto result = flowgraph->blockCreate("invalid1", "nonexistent_type", {}, {});
        REQUIRE(result == Result::ERROR);
    }
}

TEST_CASE_METHOD(FlowgraphFixture, "Block connection", "[flowgraph]") {
    REQUIRE(flowgraph->blockCreate("gen1", "signal_generator", {}, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->blockCreate("gen2", "signal_generator", {}, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->blockCreate("gen3", "signal_generator", {}, {}) == Result::SUCCESS);

    TensorMap addInputs;
    addInputs["a"].requested("gen1", "signal");
    addInputs["b"].requested("gen2", "signal");
    REQUIRE(flowgraph->blockCreate("add1", "add", {}, addInputs) == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("add1")->state() == Block::State::Created);

    SECTION("disconnect blocks") {
        auto result = flowgraph->blockDisconnect("add1", "a");
        REQUIRE((result == Result::SUCCESS or result == Result::INCOMPLETE));
        REQUIRE(flowgraph->blockList().at("add1")->state() == Block::State::Incomplete);
    }

    SECTION("reconnect after disconnect") {
        flowgraph->blockDisconnect("add1", "a");
        REQUIRE(flowgraph->blockList().at("add1")->state() == Block::State::Incomplete);

        auto result = flowgraph->blockConnect("add1", "a", "gen1", "signal");
        REQUIRE(result == Result::SUCCESS);
        REQUIRE(flowgraph->blockList().at("add1")->state() == Block::State::Created);
    }

    SECTION("reconnect to different source") {
        flowgraph->blockDisconnect("add1", "a");
        REQUIRE(flowgraph->blockList().at("add1")->state() == Block::State::Incomplete);

        auto result = flowgraph->blockConnect("add1", "a", "gen3", "signal");
        REQUIRE(result == Result::SUCCESS);
        REQUIRE(flowgraph->blockList().at("add1")->state() == Block::State::Created);
    }

    SECTION("connect to nonexistent block fails") {
        auto result = flowgraph->blockConnect("add1", "a", "nonexistent", "signal");
        REQUIRE(result == Result::ERROR);
    }

    SECTION("connect to nonexistent source port leaves block incomplete") {
        flowgraph->blockDisconnect("add1", "a");
        auto result = flowgraph->blockConnect("add1", "a", "gen1", "nonexistent");
        REQUIRE((result == Result::SUCCESS or result == Result::INCOMPLETE));
        REQUIRE(flowgraph->blockList().at("add1")->state() == Block::State::Incomplete);
    }

    SECTION("disconnect nonexistent port fails") {
        auto result = flowgraph->blockDisconnect("add1", "nonexistent");
        REQUIRE(result == Result::ERROR);
    }
}

TEST_CASE_METHOD(FlowgraphFixture, "Downstream propagation on connect", "[flowgraph][propagation]") {
    // Create chain: gen1 -> add1 -> add2
    REQUIRE(flowgraph->blockCreate("gen1", "signal_generator", {}, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->blockCreate("gen2", "signal_generator", {}, {}) == Result::SUCCESS);

    // Connect gen1 -> add1.a
    TensorMap add1Inputs;
    add1Inputs["a"].requested("gen1", "signal");
    add1Inputs["b"].requested("gen2", "signal");
    REQUIRE(flowgraph->blockCreate("add1", "add", {}, add1Inputs) == Result::SUCCESS);

    // Connect add1 -> add2.a
    TensorMap add2Inputs;
    add2Inputs["a"].requested("add1", "sum");
    add2Inputs["b"].requested("gen2", "signal");
    REQUIRE(flowgraph->blockCreate("add2", "add", {}, add2Inputs) == Result::SUCCESS);

    // All blocks should exist and be created
    REQUIRE(flowgraph->blockList().size() == 4);
    REQUIRE(flowgraph->blockList().at("gen1")->state() == Block::State::Created);
    REQUIRE(flowgraph->blockList().at("gen2")->state() == Block::State::Created);
    REQUIRE(flowgraph->blockList().at("add1")->state() == Block::State::Created);
    REQUIRE(flowgraph->blockList().at("add2")->state() == Block::State::Created);

    SECTION("disconnecting upstream marks downstream incomplete") {
        auto result = flowgraph->blockDisconnect("add1", "a");
        REQUIRE((result == Result::SUCCESS or result == Result::INCOMPLETE));

        // add1 should be incomplete, add2 should be incomplete (unresolved input)
        REQUIRE(flowgraph->blockList().at("add1")->state() == Block::State::Incomplete);
        REQUIRE(flowgraph->blockList().at("add2")->state() == Block::State::Incomplete);
    }

    SECTION("reconnecting upstream restores downstream") {
        flowgraph->blockDisconnect("add1", "a");
        REQUIRE(flowgraph->blockList().at("add1")->state() == Block::State::Incomplete);
        REQUIRE(flowgraph->blockList().at("add2")->state() == Block::State::Incomplete);

        auto result = flowgraph->blockConnect("add1", "a", "gen1", "signal");
        REQUIRE(result == Result::SUCCESS);
        REQUIRE(flowgraph->blockList().at("add1")->state() == Block::State::Created);
        REQUIRE(flowgraph->blockList().at("add2")->state() == Block::State::Created);
    }
}

TEST_CASE_METHOD(FlowgraphFixture, "Downstream propagation on destroy", "[flowgraph][propagation]") {
    // Create chain: gen1 -> add1 -> add2
    REQUIRE(flowgraph->blockCreate("gen1", "signal_generator", {}, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->blockCreate("gen2", "signal_generator", {}, {}) == Result::SUCCESS);

    TensorMap add1Inputs;
    add1Inputs["a"].requested("gen1", "signal");
    add1Inputs["b"].requested("gen2", "signal");
    REQUIRE(flowgraph->blockCreate("add1", "add", {}, add1Inputs) == Result::SUCCESS);

    TensorMap add2Inputs;
    add2Inputs["a"].requested("add1", "sum");
    add2Inputs["b"].requested("gen2", "signal");
    REQUIRE(flowgraph->blockCreate("add2", "add", {}, add2Inputs) == Result::SUCCESS);

    REQUIRE(flowgraph->blockList().at("add1")->state() == Block::State::Created);
    REQUIRE(flowgraph->blockList().at("add2")->state() == Block::State::Created);

    SECTION("destroying upstream marks downstream incomplete") {
        REQUIRE(flowgraph->blockDestroy("gen1") == Result::SUCCESS);

        REQUIRE(flowgraph->blockList().contains("add1"));
        REQUIRE(flowgraph->blockList().at("add1")->state() == Block::State::Incomplete);
        REQUIRE(flowgraph->blockList().at("add2")->state() == Block::State::Incomplete);
    }

    SECTION("destroying middle block marks downstream incomplete") {
        REQUIRE(flowgraph->blockDestroy("add1") == Result::SUCCESS);

        REQUIRE_FALSE(flowgraph->blockList().contains("add1"));
        REQUIRE(flowgraph->blockList().contains("add2"));
        REQUIRE(flowgraph->blockList().at("add2")->state() == Block::State::Incomplete);
    }

    SECTION("destroying all upstream sources marks downstream incomplete") {
        REQUIRE(flowgraph->blockDestroy("gen1") == Result::SUCCESS);
        REQUIRE(flowgraph->blockDestroy("gen2") == Result::SUCCESS);

        REQUIRE(flowgraph->blockList().at("add1")->state() == Block::State::Incomplete);
        REQUIRE(flowgraph->blockList().at("add2")->state() == Block::State::Incomplete);
    }
}

TEST_CASE_METHOD(FlowgraphFixture, "Destroy block with multi-input downstream", "[flowgraph][propagation]") {
    // Topology:
    //   gen1 ──► add0 ──┬──► add1 ──► add2
    //   gen2 ───────────┘
    //
    // add1 has two inputs: one from add0 (a), one from gen2 (b)
    // When add0 is destroyed, add1 should preserve connection to gen2

    REQUIRE(flowgraph->blockCreate("gen1", "signal_generator", {}, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->blockCreate("gen2", "signal_generator", {}, {}) == Result::SUCCESS);

    TensorMap add0Inputs;
    add0Inputs["a"].requested("gen1", "signal");
    add0Inputs["b"].requested("gen1", "signal");
    REQUIRE(flowgraph->blockCreate("add0", "add", {}, add0Inputs) == Result::SUCCESS);

    TensorMap add1Inputs;
    add1Inputs["a"].requested("add0", "sum");
    add1Inputs["b"].requested("gen2", "signal");
    REQUIRE(flowgraph->blockCreate("add1", "add", {}, add1Inputs) == Result::SUCCESS);

    TensorMap add2Inputs;
    add2Inputs["a"].requested("add1", "sum");
    add2Inputs["b"].requested("gen2", "signal");
    REQUIRE(flowgraph->blockCreate("add2", "add", {}, add2Inputs) == Result::SUCCESS);

    REQUIRE(flowgraph->blockList().size() == 5);
    REQUIRE(flowgraph->blockList().at("add0")->state() == Block::State::Created);
    REQUIRE(flowgraph->blockList().at("add1")->state() == Block::State::Created);
    REQUIRE(flowgraph->blockList().at("add2")->state() == Block::State::Created);

    SECTION("destroying middle block preserves other connections in downstream") {
        REQUIRE(flowgraph->blockDestroy("add0") == Result::SUCCESS);

        // add0 should be gone
        REQUIRE_FALSE(flowgraph->blockList().contains("add0"));

        // add1 should exist but be incomplete (lost a from add0)
        REQUIRE(flowgraph->blockList().contains("add1"));
        REQUIRE(flowgraph->blockList().at("add1")->state() == Block::State::Incomplete);

        // add2 should exist but be incomplete (add1 is incomplete so no valid output)
        REQUIRE(flowgraph->blockList().contains("add2"));
        REQUIRE(flowgraph->blockList().at("add2")->state() == Block::State::Incomplete);

        // gen1 and gen2 should still be created
        REQUIRE(flowgraph->blockList().at("gen1")->state() == Block::State::Created);
        REQUIRE(flowgraph->blockList().at("gen2")->state() == Block::State::Created);
    }

    SECTION("reconnecting after middle block destruction restores chain") {
        REQUIRE(flowgraph->blockDestroy("add0") == Result::SUCCESS);

        // Reconnect add1.a to gen1 directly
        REQUIRE(flowgraph->blockConnect("add1", "a", "gen1", "signal") == Result::SUCCESS);

        // Both add1 and add2 should now be Created
        REQUIRE(flowgraph->blockList().at("add1")->state() == Block::State::Created);
        REQUIRE(flowgraph->blockList().at("add2")->state() == Block::State::Created);
    }
}

TEST_CASE_METHOD(FlowgraphFixture, "Destroy block with deep downstream chain", "[flowgraph][propagation]") {
    // Topology: gen1 -> add1 -> add2 -> add3 -> add4
    // Destroying add1 should leave add2, add3, add4 all incomplete

    REQUIRE(flowgraph->blockCreate("gen1", "signal_generator", {}, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->blockCreate("gen2", "signal_generator", {}, {}) == Result::SUCCESS);

    TensorMap add1Inputs;
    add1Inputs["a"].requested("gen1", "signal");
    add1Inputs["b"].requested("gen2", "signal");
    REQUIRE(flowgraph->blockCreate("add1", "add", {}, add1Inputs) == Result::SUCCESS);

    TensorMap add2Inputs;
    add2Inputs["a"].requested("add1", "sum");
    add2Inputs["b"].requested("gen2", "signal");
    REQUIRE(flowgraph->blockCreate("add2", "add", {}, add2Inputs) == Result::SUCCESS);

    TensorMap add3Inputs;
    add3Inputs["a"].requested("add2", "sum");
    add3Inputs["b"].requested("gen2", "signal");
    REQUIRE(flowgraph->blockCreate("add3", "add", {}, add3Inputs) == Result::SUCCESS);

    TensorMap add4Inputs;
    add4Inputs["a"].requested("add3", "sum");
    add4Inputs["b"].requested("gen2", "signal");
    REQUIRE(flowgraph->blockCreate("add4", "add", {}, add4Inputs) == Result::SUCCESS);

    REQUIRE(flowgraph->blockList().size() == 6);

    SECTION("destroying early block marks entire downstream chain incomplete") {
        REQUIRE(flowgraph->blockDestroy("add1") == Result::SUCCESS);

        REQUIRE_FALSE(flowgraph->blockList().contains("add1"));
        REQUIRE(flowgraph->blockList().at("add2")->state() == Block::State::Incomplete);
        REQUIRE(flowgraph->blockList().at("add3")->state() == Block::State::Incomplete);
        REQUIRE(flowgraph->blockList().at("add4")->state() == Block::State::Incomplete);
    }

    SECTION("destroying middle of chain marks only downstream incomplete") {
        REQUIRE(flowgraph->blockDestroy("add2") == Result::SUCCESS);

        REQUIRE(flowgraph->blockList().at("add1")->state() == Block::State::Created);
        REQUIRE_FALSE(flowgraph->blockList().contains("add2"));
        REQUIRE(flowgraph->blockList().at("add3")->state() == Block::State::Incomplete);
        REQUIRE(flowgraph->blockList().at("add4")->state() == Block::State::Incomplete);
    }

    SECTION("destroying end of chain leaves upstream created") {
        REQUIRE(flowgraph->blockDestroy("add4") == Result::SUCCESS);

        REQUIRE(flowgraph->blockList().at("add1")->state() == Block::State::Created);
        REQUIRE(flowgraph->blockList().at("add2")->state() == Block::State::Created);
        REQUIRE(flowgraph->blockList().at("add3")->state() == Block::State::Created);
        REQUIRE_FALSE(flowgraph->blockList().contains("add4"));
    }
}

TEST_CASE_METHOD(FlowgraphFixture, "Destroy block with diamond dependency", "[flowgraph][propagation]") {
    // Topology (diamond):
    //        ┌──► add1 ──┐
    // gen1 ──┤           ├──► add3
    //        └──► add2 ──┘
    //
    // add3 depends on both add1 and add2

    REQUIRE(flowgraph->blockCreate("gen1", "signal_generator", {}, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->blockCreate("gen2", "signal_generator", {}, {}) == Result::SUCCESS);

    TensorMap add1Inputs;
    add1Inputs["a"].requested("gen1", "signal");
    add1Inputs["b"].requested("gen2", "signal");
    REQUIRE(flowgraph->blockCreate("add1", "add", {}, add1Inputs) == Result::SUCCESS);

    TensorMap add2Inputs;
    add2Inputs["a"].requested("gen1", "signal");
    add2Inputs["b"].requested("gen2", "signal");
    REQUIRE(flowgraph->blockCreate("add2", "add", {}, add2Inputs) == Result::SUCCESS);

    TensorMap add3Inputs;
    add3Inputs["a"].requested("add1", "sum");
    add3Inputs["b"].requested("add2", "sum");
    REQUIRE(flowgraph->blockCreate("add3", "add", {}, add3Inputs) == Result::SUCCESS);

    REQUIRE(flowgraph->blockList().size() == 5);
    REQUIRE(flowgraph->blockList().at("add3")->state() == Block::State::Created);

    SECTION("destroying one branch leaves add3 incomplete but preserves other branch") {
        REQUIRE(flowgraph->blockDestroy("add1") == Result::SUCCESS);

        REQUIRE_FALSE(flowgraph->blockList().contains("add1"));
        REQUIRE(flowgraph->blockList().at("add2")->state() == Block::State::Created);
        REQUIRE(flowgraph->blockList().at("add3")->state() == Block::State::Incomplete);
    }

    SECTION("destroying other branch leaves add3 incomplete but preserves first branch") {
        REQUIRE(flowgraph->blockDestroy("add2") == Result::SUCCESS);

        REQUIRE(flowgraph->blockList().at("add1")->state() == Block::State::Created);
        REQUIRE_FALSE(flowgraph->blockList().contains("add2"));
        REQUIRE(flowgraph->blockList().at("add3")->state() == Block::State::Incomplete);
    }

    SECTION("destroying source marks all downstream incomplete") {
        REQUIRE(flowgraph->blockDestroy("gen1") == Result::SUCCESS);

        REQUIRE_FALSE(flowgraph->blockList().contains("gen1"));
        REQUIRE(flowgraph->blockList().at("add1")->state() == Block::State::Incomplete);
        REQUIRE(flowgraph->blockList().at("add2")->state() == Block::State::Incomplete);
        REQUIRE(flowgraph->blockList().at("add3")->state() == Block::State::Incomplete);
    }

    SECTION("reconnecting both branches restores add3") {
        REQUIRE(flowgraph->blockDestroy("add1") == Result::SUCCESS);
        REQUIRE(flowgraph->blockList().at("add3")->state() == Block::State::Incomplete);

        // Reconnect add3.a to gen1 directly (bypassing removed add1)
        REQUIRE(flowgraph->blockConnect("add3", "a", "gen1", "signal") == Result::SUCCESS);
        REQUIRE(flowgraph->blockList().at("add3")->state() == Block::State::Created);
    }
}

TEST_CASE_METHOD(FlowgraphFixture, "Destroy block with multiple dependents at same level", "[flowgraph][propagation]") {
    // Topology:
    //        ┌──► add1
    // add0 ──┼──► add2
    //        └──► add3
    //
    // add0 fans out to three blocks

    REQUIRE(flowgraph->blockCreate("gen1", "signal_generator", {}, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->blockCreate("gen2", "signal_generator", {}, {}) == Result::SUCCESS);

    TensorMap add0Inputs;
    add0Inputs["a"].requested("gen1", "signal");
    add0Inputs["b"].requested("gen2", "signal");
    REQUIRE(flowgraph->blockCreate("add0", "add", {}, add0Inputs) == Result::SUCCESS);

    TensorMap add1Inputs;
    add1Inputs["a"].requested("add0", "sum");
    add1Inputs["b"].requested("gen2", "signal");
    REQUIRE(flowgraph->blockCreate("add1", "add", {}, add1Inputs) == Result::SUCCESS);

    TensorMap add2Inputs;
    add2Inputs["a"].requested("add0", "sum");
    add2Inputs["b"].requested("gen2", "signal");
    REQUIRE(flowgraph->blockCreate("add2", "add", {}, add2Inputs) == Result::SUCCESS);

    TensorMap add3Inputs;
    add3Inputs["a"].requested("add0", "sum");
    add3Inputs["b"].requested("gen2", "signal");
    REQUIRE(flowgraph->blockCreate("add3", "add", {}, add3Inputs) == Result::SUCCESS);

    REQUIRE(flowgraph->blockList().size() == 6);

    SECTION("destroying fan-out source marks all dependents incomplete") {
        REQUIRE(flowgraph->blockDestroy("add0") == Result::SUCCESS);

        REQUIRE_FALSE(flowgraph->blockList().contains("add0"));
        REQUIRE(flowgraph->blockList().at("add1")->state() == Block::State::Incomplete);
        REQUIRE(flowgraph->blockList().at("add2")->state() == Block::State::Incomplete);
        REQUIRE(flowgraph->blockList().at("add3")->state() == Block::State::Incomplete);

        // gen1 and gen2 should still be created
        REQUIRE(flowgraph->blockList().at("gen1")->state() == Block::State::Created);
        REQUIRE(flowgraph->blockList().at("gen2")->state() == Block::State::Created);
    }
}

TEST_CASE_METHOD(FlowgraphFixture, "Destroy with no propagation flag", "[flowgraph][propagation]") {
    // Test propagate=false behavior

    REQUIRE(flowgraph->blockCreate("gen1", "signal_generator", {}, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->blockCreate("gen2", "signal_generator", {}, {}) == Result::SUCCESS);

    TensorMap add1Inputs;
    add1Inputs["a"].requested("gen1", "signal");
    add1Inputs["b"].requested("gen2", "signal");
    REQUIRE(flowgraph->blockCreate("add1", "add", {}, add1Inputs) == Result::SUCCESS);

    TensorMap add2Inputs;
    add2Inputs["a"].requested("add1", "sum");
    add2Inputs["b"].requested("gen2", "signal");
    REQUIRE(flowgraph->blockCreate("add2", "add", {}, add2Inputs) == Result::SUCCESS);

    SECTION("destroy with propagate=false does not touch downstream") {
        // This is used internally during cascade operations
        // Downstream blocks remain but may have stale references (internal use only)
        REQUIRE(flowgraph->blockDestroy("add1", false) == Result::SUCCESS);

        REQUIRE_FALSE(flowgraph->blockList().contains("add1"));
        // add2 still exists (not recreated, may have invalid state - internal use)
        REQUIRE(flowgraph->blockList().contains("add2"));
    }
}

TEST_CASE_METHOD(FlowgraphFixture, "Destroy block then recreate with same name", "[flowgraph][propagation]") {
    REQUIRE(flowgraph->blockCreate("gen1", "signal_generator", {}, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->blockCreate("gen2", "signal_generator", {}, {}) == Result::SUCCESS);

    TensorMap add1Inputs;
    add1Inputs["a"].requested("gen1", "signal");
    add1Inputs["b"].requested("gen2", "signal");
    REQUIRE(flowgraph->blockCreate("add1", "add", {}, add1Inputs) == Result::SUCCESS);

    TensorMap add2Inputs;
    add2Inputs["a"].requested("add1", "sum");
    add2Inputs["b"].requested("gen2", "signal");
    REQUIRE(flowgraph->blockCreate("add2", "add", {}, add2Inputs) == Result::SUCCESS);

    SECTION("can recreate block with same name after destruction") {
        REQUIRE(flowgraph->blockDestroy("add1") == Result::SUCCESS);
        REQUIRE_FALSE(flowgraph->blockList().contains("add1"));
        REQUIRE(flowgraph->blockList().at("add2")->state() == Block::State::Incomplete);

        // Recreate add1
        REQUIRE(flowgraph->blockCreate("add1", "add", {}, add1Inputs) == Result::SUCCESS);
        REQUIRE(flowgraph->blockList().at("add1")->state() == Block::State::Created);

        // Reconnect add2 to the new add1
        REQUIRE(flowgraph->blockConnect("add2", "a", "add1", "sum") == Result::SUCCESS);
        REQUIRE(flowgraph->blockList().at("add2")->state() == Block::State::Created);
    }
}

TEST_CASE_METHOD(FlowgraphFixture, "Destroy source block used by multiple inputs of same block", "[flowgraph][propagation]") {
    // Topology: gen1 feeds both inputs of add1
    // gen1 ──┬──► add1.a
    //        └──► add1.b

    REQUIRE(flowgraph->blockCreate("gen1", "signal_generator", {}, {}) == Result::SUCCESS);

    TensorMap add1Inputs;
    add1Inputs["a"].requested("gen1", "signal");
    add1Inputs["b"].requested("gen1", "signal");
    REQUIRE(flowgraph->blockCreate("add1", "add", {}, add1Inputs) == Result::SUCCESS);

    REQUIRE(flowgraph->blockList().at("add1")->state() == Block::State::Created);

    SECTION("destroying source disconnects all inputs from that source") {
        REQUIRE(flowgraph->blockDestroy("gen1") == Result::SUCCESS);

        REQUIRE_FALSE(flowgraph->blockList().contains("gen1"));
        REQUIRE(flowgraph->blockList().contains("add1"));
        REQUIRE(flowgraph->blockList().at("add1")->state() == Block::State::Incomplete);
    }
}

TEST_CASE_METHOD(FlowgraphFixture, "Fan-out propagation", "[flowgraph][propagation]") {
    // gen1 -> add1
    // gen1 -> add2
    REQUIRE(flowgraph->blockCreate("gen1", "signal_generator", {}, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->blockCreate("gen2", "signal_generator", {}, {}) == Result::SUCCESS);

    TensorMap add1Inputs;
    add1Inputs["a"].requested("gen1", "signal");
    add1Inputs["b"].requested("gen2", "signal");
    REQUIRE(flowgraph->blockCreate("add1", "add", {}, add1Inputs) == Result::SUCCESS);

    TensorMap add2Inputs;
    add2Inputs["a"].requested("gen1", "signal");
    add2Inputs["b"].requested("gen2", "signal");
    REQUIRE(flowgraph->blockCreate("add2", "add", {}, add2Inputs) == Result::SUCCESS);

    REQUIRE(flowgraph->blockList().at("add1")->state() == Block::State::Created);
    REQUIRE(flowgraph->blockList().at("add2")->state() == Block::State::Created);

    SECTION("destroying shared source marks all downstream incomplete") {
        REQUIRE(flowgraph->blockDestroy("gen1") == Result::SUCCESS);

        REQUIRE(flowgraph->blockList().at("add1")->state() == Block::State::Incomplete);
        REQUIRE(flowgraph->blockList().at("add2")->state() == Block::State::Incomplete);
    }

    SECTION("disconnecting from shared source marks only that block incomplete") {
        flowgraph->blockDisconnect("add1", "a");

        REQUIRE(flowgraph->blockList().at("add1")->state() == Block::State::Incomplete);
        REQUIRE(flowgraph->blockList().at("add2")->state() == Block::State::Created);
    }
}

TEST_CASE_METHOD(FlowgraphFixture, "Block reconfiguration", "[flowgraph]") {
    REQUIRE(flowgraph->blockCreate("gen1", "signal_generator", {}, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("gen1")->state() == Block::State::Created);

    SECTION("reconfigure existing block") {
        Parser::Map newConfig;
        newConfig["bufferSize"] = std::string("2048");
        auto result = flowgraph->blockReconfigure("gen1", newConfig);
        REQUIRE(result == Result::SUCCESS);
        REQUIRE(flowgraph->blockList().at("gen1")->state() == Block::State::Created);
    }

    SECTION("reconfigure nonexistent block fails") {
        Parser::Map newConfig;
        auto result = flowgraph->blockReconfigure("nonexistent", newConfig);
        REQUIRE(result == Result::ERROR);
    }
}

TEST_CASE_METHOD(FlowgraphFixture, "Block recreation", "[flowgraph][recreation]") {
    REQUIRE(flowgraph->blockCreate("gen1", "signal_generator", {}, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("gen1")->state() == Block::State::Created);

    SECTION("recreate single block") {
        Parser::Map newConfig;
        newConfig["bufferSize"] = std::string("4096");
        auto result = flowgraph->blockRecreate("gen1", newConfig);
        REQUIRE(result == Result::SUCCESS);
        REQUIRE(flowgraph->blockList().size() == 1);
        REQUIRE(flowgraph->blockList().contains("gen1"));
        REQUIRE(flowgraph->blockList().at("gen1")->state() == Block::State::Created);
    }

    SECTION("recreate nonexistent block fails") {
        Parser::Map newConfig;
        auto result = flowgraph->blockRecreate("nonexistent", newConfig);
        REQUIRE(result == Result::ERROR);
    }
}

TEST_CASE_METHOD(FlowgraphFixture, "Block recreation can change implementation", "[flowgraph][recreation]") {
    REQUIRE(RegisterSignalGeneratorTestProvider() == Result::SUCCESS);
    REQUIRE(flowgraph->blockCreate("gen1", "signal_generator", {}, {}) == Result::SUCCESS);

    Parser::Map config;
    REQUIRE(flowgraph->blockConfig("gen1", config) == Result::SUCCESS);

    const auto& block = flowgraph->blockList().at("gen1");
    const auto implementations = Registry::ListAvailableModules("signal_generator",
                                                                 block->device(),
                                                                 block->runtime(),
                                                                 kSignalGeneratorTestProvider);

    REQUIRE(implementations.size() == 1);
    REQUIRE(block->provider() != implementations.front().provider);

    REQUIRE(flowgraph->blockRecreate("gen1",
                                     config,
                                     implementations.front().device,
                                     implementations.front().runtime,
                                     implementations.front().provider) == Result::SUCCESS);

    const auto& recreated = flowgraph->blockList().at("gen1");
    REQUIRE(recreated->state() == Block::State::Created);
    REQUIRE(recreated->device() == implementations.front().device);
    REQUIRE(recreated->runtime() == implementations.front().runtime);
    REQUIRE(recreated->provider() == implementations.front().provider);
}

TEST_CASE_METHOD(FlowgraphFixture, "Block recreation with downstream chain", "[flowgraph][recreation]") {
    // Topology: gen1 -> add1 -> add2 -> add3
    REQUIRE(flowgraph->blockCreate("gen1", "signal_generator", {}, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->blockCreate("gen2", "signal_generator", {}, {}) == Result::SUCCESS);

    TensorMap add1Inputs;
    add1Inputs["a"].requested("gen1", "signal");
    add1Inputs["b"].requested("gen2", "signal");
    REQUIRE(flowgraph->blockCreate("add1", "add", {}, add1Inputs) == Result::SUCCESS);

    TensorMap add2Inputs;
    add2Inputs["a"].requested("add1", "sum");
    add2Inputs["b"].requested("gen2", "signal");
    REQUIRE(flowgraph->blockCreate("add2", "add", {}, add2Inputs) == Result::SUCCESS);

    TensorMap add3Inputs;
    add3Inputs["a"].requested("add2", "sum");
    add3Inputs["b"].requested("gen2", "signal");
    REQUIRE(flowgraph->blockCreate("add3", "add", {}, add3Inputs) == Result::SUCCESS);

    REQUIRE(flowgraph->blockList().size() == 5);
    REQUIRE(flowgraph->blockList().at("add1")->state() == Block::State::Created);
    REQUIRE(flowgraph->blockList().at("add2")->state() == Block::State::Created);
    REQUIRE(flowgraph->blockList().at("add3")->state() == Block::State::Created);

    SECTION("recreating source recreates entire downstream chain") {
        Parser::Map newConfig;
        auto result = flowgraph->blockRecreate("gen1", newConfig);
        REQUIRE(result == Result::SUCCESS);

        // All blocks should still exist and be created
        REQUIRE(flowgraph->blockList().size() == 5);
        REQUIRE(flowgraph->blockList().at("gen1")->state() == Block::State::Created);
        REQUIRE(flowgraph->blockList().at("gen2")->state() == Block::State::Created);
        REQUIRE(flowgraph->blockList().at("add1")->state() == Block::State::Created);
        REQUIRE(flowgraph->blockList().at("add2")->state() == Block::State::Created);
        REQUIRE(flowgraph->blockList().at("add3")->state() == Block::State::Created);
    }

    SECTION("recreating middle block recreates downstream only") {
        Parser::Map newConfig;
        auto result = flowgraph->blockRecreate("add1", newConfig);
        REQUIRE(result == Result::SUCCESS);

        // All blocks should still exist and be created
        REQUIRE(flowgraph->blockList().size() == 5);
        REQUIRE(flowgraph->blockList().at("gen1")->state() == Block::State::Created);
        REQUIRE(flowgraph->blockList().at("gen2")->state() == Block::State::Created);
        REQUIRE(flowgraph->blockList().at("add1")->state() == Block::State::Created);
        REQUIRE(flowgraph->blockList().at("add2")->state() == Block::State::Created);
        REQUIRE(flowgraph->blockList().at("add3")->state() == Block::State::Created);
    }

    SECTION("recreating end block recreates only that block") {
        Parser::Map newConfig;
        auto result = flowgraph->blockRecreate("add3", newConfig);
        REQUIRE(result == Result::SUCCESS);

        // All blocks should still exist and be created
        REQUIRE(flowgraph->blockList().size() == 5);
        REQUIRE(flowgraph->blockList().at("add1")->state() == Block::State::Created);
        REQUIRE(flowgraph->blockList().at("add2")->state() == Block::State::Created);
        REQUIRE(flowgraph->blockList().at("add3")->state() == Block::State::Created);
    }
}

TEST_CASE_METHOD(FlowgraphFixture, "Block recreation with diamond dependency", "[flowgraph][recreation]") {
    // Topology (diamond):
    //        ┌──► add1 ──┐
    // gen1 ──┤           ├──► add3
    //        └──► add2 ──┘

    REQUIRE(flowgraph->blockCreate("gen1", "signal_generator", {}, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->blockCreate("gen2", "signal_generator", {}, {}) == Result::SUCCESS);

    TensorMap add1Inputs;
    add1Inputs["a"].requested("gen1", "signal");
    add1Inputs["b"].requested("gen2", "signal");
    REQUIRE(flowgraph->blockCreate("add1", "add", {}, add1Inputs) == Result::SUCCESS);

    TensorMap add2Inputs;
    add2Inputs["a"].requested("gen1", "signal");
    add2Inputs["b"].requested("gen2", "signal");
    REQUIRE(flowgraph->blockCreate("add2", "add", {}, add2Inputs) == Result::SUCCESS);

    TensorMap add3Inputs;
    add3Inputs["a"].requested("add1", "sum");
    add3Inputs["b"].requested("add2", "sum");
    REQUIRE(flowgraph->blockCreate("add3", "add", {}, add3Inputs) == Result::SUCCESS);

    REQUIRE(flowgraph->blockList().size() == 5);
    REQUIRE(flowgraph->blockList().at("add3")->state() == Block::State::Created);

    SECTION("recreating source recreates all downstream including diamond") {
        Parser::Map newConfig;
        auto result = flowgraph->blockRecreate("gen1", newConfig);
        REQUIRE(result == Result::SUCCESS);

        REQUIRE(flowgraph->blockList().size() == 5);
        REQUIRE(flowgraph->blockList().at("gen1")->state() == Block::State::Created);
        REQUIRE(flowgraph->blockList().at("add1")->state() == Block::State::Created);
        REQUIRE(flowgraph->blockList().at("add2")->state() == Block::State::Created);
        REQUIRE(flowgraph->blockList().at("add3")->state() == Block::State::Created);
    }

    SECTION("recreating one branch recreates that branch and convergence point") {
        Parser::Map newConfig;
        auto result = flowgraph->blockRecreate("add1", newConfig);
        REQUIRE(result == Result::SUCCESS);

        REQUIRE(flowgraph->blockList().size() == 5);
        REQUIRE(flowgraph->blockList().at("add1")->state() == Block::State::Created);
        REQUIRE(flowgraph->blockList().at("add2")->state() == Block::State::Created);
        REQUIRE(flowgraph->blockList().at("add3")->state() == Block::State::Created);
    }
}

TEST_CASE_METHOD(FlowgraphFixture, "Block recreation with fan-out", "[flowgraph][recreation]") {
    // Topology:
    //        ┌──► add1
    // add0 ──┼──► add2
    //        └──► add3

    REQUIRE(flowgraph->blockCreate("gen1", "signal_generator", {}, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->blockCreate("gen2", "signal_generator", {}, {}) == Result::SUCCESS);

    TensorMap add0Inputs;
    add0Inputs["a"].requested("gen1", "signal");
    add0Inputs["b"].requested("gen2", "signal");
    REQUIRE(flowgraph->blockCreate("add0", "add", {}, add0Inputs) == Result::SUCCESS);

    TensorMap add1Inputs;
    add1Inputs["a"].requested("add0", "sum");
    add1Inputs["b"].requested("gen2", "signal");
    REQUIRE(flowgraph->blockCreate("add1", "add", {}, add1Inputs) == Result::SUCCESS);

    TensorMap add2Inputs;
    add2Inputs["a"].requested("add0", "sum");
    add2Inputs["b"].requested("gen2", "signal");
    REQUIRE(flowgraph->blockCreate("add2", "add", {}, add2Inputs) == Result::SUCCESS);

    TensorMap add3Inputs;
    add3Inputs["a"].requested("add0", "sum");
    add3Inputs["b"].requested("gen2", "signal");
    REQUIRE(flowgraph->blockCreate("add3", "add", {}, add3Inputs) == Result::SUCCESS);

    REQUIRE(flowgraph->blockList().size() == 6);

    SECTION("recreating fan-out source recreates all dependents") {
        Parser::Map newConfig;
        auto result = flowgraph->blockRecreate("add0", newConfig);
        REQUIRE(result == Result::SUCCESS);

        REQUIRE(flowgraph->blockList().size() == 6);
        REQUIRE(flowgraph->blockList().at("add0")->state() == Block::State::Created);
        REQUIRE(flowgraph->blockList().at("add1")->state() == Block::State::Created);
        REQUIRE(flowgraph->blockList().at("add2")->state() == Block::State::Created);
        REQUIRE(flowgraph->blockList().at("add3")->state() == Block::State::Created);
    }

    SECTION("recreating one fan-out target does not affect siblings") {
        Parser::Map newConfig;
        auto result = flowgraph->blockRecreate("add1", newConfig);
        REQUIRE(result == Result::SUCCESS);

        REQUIRE(flowgraph->blockList().size() == 6);
        REQUIRE(flowgraph->blockList().at("add0")->state() == Block::State::Created);
        REQUIRE(flowgraph->blockList().at("add1")->state() == Block::State::Created);
        REQUIRE(flowgraph->blockList().at("add2")->state() == Block::State::Created);
        REQUIRE(flowgraph->blockList().at("add3")->state() == Block::State::Created);
    }
}

TEST_CASE_METHOD(FlowgraphFixture, "Block recreation preserves connections", "[flowgraph][recreation]") {
    // Verify that after recreation, all connections are preserved

    REQUIRE(flowgraph->blockCreate("gen1", "signal_generator", {}, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->blockCreate("gen2", "signal_generator", {}, {}) == Result::SUCCESS);

    TensorMap add1Inputs;
    add1Inputs["a"].requested("gen1", "signal");
    add1Inputs["b"].requested("gen2", "signal");
    REQUIRE(flowgraph->blockCreate("add1", "add", {}, add1Inputs) == Result::SUCCESS);

    TensorMap add2Inputs;
    add2Inputs["a"].requested("add1", "sum");
    add2Inputs["b"].requested("gen2", "signal");
    REQUIRE(flowgraph->blockCreate("add2", "add", {}, add2Inputs) == Result::SUCCESS);

    SECTION("connections are preserved after recreation") {
        Parser::Map newConfig;
        REQUIRE(flowgraph->blockRecreate("add1", newConfig) == Result::SUCCESS);

        // All blocks should be created, meaning connections are valid
        REQUIRE(flowgraph->blockList().at("add1")->state() == Block::State::Created);
        REQUIRE(flowgraph->blockList().at("add2")->state() == Block::State::Created);

        // Verify add2 still has its connection to add1
        const auto& add2Ptr = flowgraph->blockList().at("add2");
        const auto& inputs = add2Ptr->inputs();
        REQUIRE(inputs.contains("a"));
        REQUIRE(inputs.at("a").external.has_value());
        REQUIRE(inputs.at("a").external->block == "add1");
        REQUIRE(inputs.at("a").external->port == "sum");
    }
}

TEST_CASE_METHOD(FlowgraphFixture, "Flowgraph serialization", "[flowgraph][serialization]") {
    REQUIRE(flowgraph->blockCreate("gen1", "signal_generator", {}, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->blockCreate("gen2", "signal_generator", {}, {}) == Result::SUCCESS);

    TensorMap addInputs;
    addInputs["a"].requested("gen1", "signal");
    addInputs["b"].requested("gen2", "signal");
    REQUIRE(flowgraph->blockCreate("add1", "add", {}, addInputs) == Result::SUCCESS);

    REQUIRE(flowgraph->blockList().at("gen1")->state() == Block::State::Created);
    REQUIRE(flowgraph->blockList().at("gen2")->state() == Block::State::Created);
    REQUIRE(flowgraph->blockList().at("add1")->state() == Block::State::Created);

    flowgraph->setTitle("Test Flowgraph");
    flowgraph->setAuthor("Test Author");

    StackDockLayoutFixture graphLeaf;
    graphLeaf.flowgraphs.push_back({.order = 0});

    StackDockLayoutFixture spectrumLeaf;
    spectrumLeaf.surfaces.push_back({
        .block = "add1",
        .surface = "spectrum",
        .order = 0,
    });

    StackDockLayoutFixture stackRoot;
    stackRoot.direction = "left";
    stackRoot.ratio = 0.72f;
    stackRoot.children["primary"] = graphLeaf;
    stackRoot.children["secondary"] = spectrumLeaf;

    StackDockLayoutFixture inspectorRoot;
    inspectorRoot.surfaces.push_back({
        .block = "add1",
        .surface = "waterfall",
        .order = 0,
    });

    std::unordered_map<std::string, StackMetaFixture> stackWindows;
    stackWindows["stack_0"].title = "Stack 0";
    stackWindows["stack_0"].x = 10.0f;
    stackWindows["stack_0"].y = 20.0f;
    stackWindows["stack_0"].width = 500.0f;
    stackWindows["stack_0"].height = 300.0f;
    stackWindows["stack_0"].layout["root"] = stackRoot;

    stackWindows["stack_1"].title = "Inspector";
    stackWindows["stack_1"].x = 30.0f;
    stackWindows["stack_1"].y = 40.0f;
    stackWindows["stack_1"].width = 640.0f;
    stackWindows["stack_1"].height = 360.0f;
    stackWindows["stack_1"].layout["root"] = inspectorRoot;
    REQUIRE(flowgraph->setMeta("stacks", stackWindows) == Result::SUCCESS);

    SECTION("export to blob") {
        std::vector<char> blob;
        auto result = flowgraph->exportToBlob(blob);
        REQUIRE(result == Result::SUCCESS);
        REQUIRE(!blob.empty());

        const std::string yaml(blob.begin(), blob.end());
        REQUIRE(yaml.starts_with("---\nversion: 2\n"));
        REQUIRE(yaml.find("stacks:") != std::string::npos);
        REQUIRE(yaml.find("stack_0:") != std::string::npos);
        REQUIRE(yaml.find("title: Stack 0") != std::string::npos);
        REQUIRE(yaml.find("layout:") != std::string::npos);
        REQUIRE(yaml.find("direction: left") != std::string::npos);
        REQUIRE(yaml.find("ratio: 0.72") != std::string::npos);
        REQUIRE(yaml.find("splitDirection:") == std::string::npos);
        REQUIRE(yaml.find("splitRatio:") == std::string::npos);
        REQUIRE(yaml.find("flowgraphs:") != std::string::npos);
        REQUIRE(yaml.find("surfaces:") != std::string::npos);
        REQUIRE(yaml.find("items:") == std::string::npos);
        REQUIRE(yaml.find("kind:") == std::string::npos);
        REQUIRE(yaml.find("surface:add1:spectrum") == std::string::npos);
        REQUIRE(yaml.find("block: add1") != std::string::npos);
        REQUIRE(yaml.find("surface: spectrum") != std::string::npos);
        REQUIRE(yaml.find("width: 360") == std::string::npos);
        REQUIRE(yaml.find("width: 140") == std::string::npos);
    }

    SECTION("export and reimport") {
        std::vector<char> blob;
        REQUIRE(flowgraph->exportToBlob(blob) == Result::SUCCESS);

        // Destroy current blocks
        std::vector<std::string> names;
        for (const auto& [name, _] : flowgraph->blockList()) {
            names.push_back(name);
        }
        for (const auto& name : names) {
            flowgraph->blockDestroy(name, false);
        }
        REQUIRE(flowgraph->blockList().empty());

        // Reimport
        REQUIRE(flowgraph->importFromBlob(blob) == Result::SUCCESS);
        REQUIRE(flowgraph->blockList().size() == 3);
        REQUIRE(flowgraph->blockList().at("gen1")->state() == Block::State::Created);
        REQUIRE(flowgraph->blockList().at("gen2")->state() == Block::State::Created);
        REQUIRE(flowgraph->blockList().at("add1")->state() == Block::State::Created);
        REQUIRE(flowgraph->title() == "Test Flowgraph");
        REQUIRE(flowgraph->author() == "Test Author");

        std::unordered_map<std::string, StackMetaFixture> restoredStackWindows;
        REQUIRE(flowgraph->getMeta("stacks", restoredStackWindows) == Result::SUCCESS);
        REQUIRE(restoredStackWindows.size() == 2);

        const auto& stackZero = restoredStackWindows.at("stack_0");
        const auto& inspector = restoredStackWindows.at("stack_1");
        REQUIRE(stackZero.title == "Stack 0");
        REQUIRE(stackZero.x == Catch::Approx(10.0f));
        REQUIRE(stackZero.y == Catch::Approx(20.0f));
        REQUIRE(stackZero.width == Catch::Approx(500.0f));
        REQUIRE(stackZero.height == Catch::Approx(300.0f));
        REQUIRE(stackZero.layout.size() == 1);
        REQUIRE(inspector.title == "Inspector");
        REQUIRE(inspector.x == Catch::Approx(30.0f));
        REQUIRE(inspector.y == Catch::Approx(40.0f));
        REQUIRE(inspector.width == Catch::Approx(640.0f));
        REQUIRE(inspector.height == Catch::Approx(360.0f));
        REQUIRE(inspector.layout.size() == 1);

        const auto& stackRootLayout = stackZero.layout.at("root");
        REQUIRE(stackRootLayout.direction == "left");
        REQUIRE(stackRootLayout.ratio == Catch::Approx(0.72f));
        REQUIRE(stackRootLayout.children.size() == 2);
        REQUIRE(stackRootLayout.children.contains("primary"));
        REQUIRE(stackRootLayout.children.contains("secondary"));

        const auto& restoredGraphLeaf = stackRootLayout.children.at("primary");
        REQUIRE(restoredGraphLeaf.flowgraphs.size() == 1);
        REQUIRE(restoredGraphLeaf.flowgraphs.at(0).order == 0);

        const auto& restoredSpectrumLeaf = stackRootLayout.children.at("secondary");
        REQUIRE(restoredSpectrumLeaf.surfaces.size() == 1);
        REQUIRE(restoredSpectrumLeaf.surfaces.at(0).block == "add1");
        REQUIRE(restoredSpectrumLeaf.surfaces.at(0).surface == "spectrum");

        const auto& inspectorLayout = inspector.layout.at("root");
        REQUIRE(inspectorLayout.surfaces.size() == 1);
        REQUIRE(inspectorLayout.surfaces.at(0).block == "add1");
        REQUIRE(inspectorLayout.surfaces.at(0).surface == "waterfall");
    }
}

int main(int argc, char* argv[]) {
    JST_LOG_SET_DEBUG_LEVEL(0);
    return Catch::Session().run(argc, argv);
}
