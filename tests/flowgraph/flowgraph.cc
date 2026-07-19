#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "common.hh"

#include "jetstream/detail/block_impl.hh"

namespace {

using namespace Jetstream;

struct RuntimeErrorTestState {
    U64 calls = 0;
    U64 sinkCalls = 0;

    void reset() {
        calls = 0;
        sinkCalls = 0;
    }
};

RuntimeErrorTestState& runtimeErrorTestState() {
    static RuntimeErrorTestState state;
    return state;
}

struct RuntimeErrorTestModuleConfig : Module::Config {
    JST_MODULE_TYPE(runtime_error_test_module)

    Result serialize(Parser::Map&) const override {
        return Result::SUCCESS;
    }

    Result deserialize(const Parser::Map&) override {
        return Result::SUCCESS;
    }

    std::size_t hash() const override {
        return 0;
    }
};

struct RuntimeErrorTestModule : Module::Impl,
                                DynamicConfig<RuntimeErrorTestModuleConfig>,
                                NativeCpuRuntimeContext,
                                Scheduler::Context {
    Result define() override {
        return defineInterfaceOutput("out");
    }

    Result create() override {
        JST_CHECK(output.create(DeviceType::CPU, DataType::F32, {1}));
        output.at<F32>(0) = 1.0f;
        outputs()["out"].produced(name(), "out", output);
        return Result::SUCCESS;
    }

    Result computeSubmit() override {
        runtimeErrorTestState().calls += 1;
        JST_ERROR("[RUNTIME_ERROR_TEST] Forced runtime failure.");
        return Result::ERROR;
    }

    Tensor output;
};

struct RuntimeErrorSinkTestModuleConfig : Module::Config {
    JST_MODULE_TYPE(runtime_error_sink_test_module)

    Result serialize(Parser::Map&) const override {
        return Result::SUCCESS;
    }

    Result deserialize(const Parser::Map&) override {
        return Result::SUCCESS;
    }

    std::size_t hash() const override {
        return 0;
    }
};

struct RuntimeErrorSinkTestModule : Module::Impl,
                                    DynamicConfig<RuntimeErrorSinkTestModuleConfig>,
                                    NativeCpuRuntimeContext,
                                    Scheduler::Context {
    Result define() override {
        return defineInterfaceInput("in");
    }

    Result computeSubmit() override {
        runtimeErrorTestState().sinkCalls += 1;
        return Result::SUCCESS;
    }
};

struct RuntimeErrorTestBlockConfig : Block::Config {
    JST_BLOCK_TYPE(runtime_error_test)
    JST_BLOCK_DOMAIN("test")
    JST_BLOCK_DESCRIPTION("Runtime Error Test",
                          "Fails during compute.",
                          "A test-only block used to verify runtime module failures mark blocks errored.")

    Result serialize(Parser::Map&) const override {
        return Result::SUCCESS;
    }

    Result deserialize(const Parser::Map&) override {
        return Result::SUCCESS;
    }

    std::size_t hash() const override {
        return 0;
    }
};

struct RuntimeErrorTestBlock : Block::Impl,
                               DynamicConfig<RuntimeErrorTestBlockConfig> {
    Result define() override {
        return defineInterfaceOutput("out", "Output", "Output tensor.");
    }

    Result create() override {
        const auto moduleConfig = std::make_shared<RuntimeErrorTestModuleConfig>();
        JST_CHECK(moduleCreate("source", moduleConfig, {}));
        JST_CHECK(moduleExposeOutput("out", {"source", "out"}));
        return Result::SUCCESS;
    }
};

struct RuntimeErrorSinkTestBlockConfig : Block::Config {
    JST_BLOCK_TYPE(runtime_error_sink_test)
    JST_BLOCK_DOMAIN("test")
    JST_BLOCK_DESCRIPTION("Runtime Error Sink Test",
                          "Consumes test output.",
                          "A test-only sink block used to verify runtime failures invalidate downstream blocks.")

    Result serialize(Parser::Map&) const override {
        return Result::SUCCESS;
    }

    Result deserialize(const Parser::Map&) override {
        return Result::SUCCESS;
    }

    std::size_t hash() const override {
        return 0;
    }
};

struct RuntimeErrorSinkTestBlock : Block::Impl,
                                   DynamicConfig<RuntimeErrorSinkTestBlockConfig> {
    Result define() override {
        return defineInterfaceInput("in", "Input", "Input tensor.");
    }

    Result create() override {
        const auto moduleConfig = std::make_shared<RuntimeErrorSinkTestModuleConfig>();
        JST_CHECK(moduleCreate("sink", moduleConfig, {{"in", inputs().at("in")}}));
        return Result::SUCCESS;
    }
};

struct EnvironmentGateTestState {
    U64 createCalls = 0;
    U64 sinkDefineCalls = 0;

    void reset() {
        createCalls = 0;
        sinkDefineCalls = 0;
    }
};

EnvironmentGateTestState& environmentGateTestState() {
    static EnvironmentGateTestState state;
    return state;
}

struct EnvironmentGateTestModuleConfig : Module::Config {
    JST_MODULE_TYPE(environment_gate_test_module)

    Result serialize(Parser::Map&) const override {
        return Result::SUCCESS;
    }

    Result deserialize(const Parser::Map&) override {
        return Result::SUCCESS;
    }

    std::size_t hash() const override {
        return 0;
    }
};

struct EnvironmentGateTestModule : Module::Impl,
                                   DynamicConfig<EnvironmentGateTestModuleConfig>,
                                   NativeCpuRuntimeContext,
                                   Scheduler::Context {
    Result define() override {
        return defineInterfaceOutput("out");
    }

    Result create() override {
        JST_CHECK(output.create(DeviceType::CPU, DataType::F32, {1}));
        output.at<F32>(0) = 1.0f;
        outputs()["out"].produced(name(), "out", output);
        return Result::SUCCESS;
    }

    Result computeSubmit() override {
        return Result::SUCCESS;
    }

    Tensor output;
};

struct EnvironmentGateTestBlockConfig : Block::Config {
    JST_BLOCK_TYPE(environment_gate_test)
    JST_BLOCK_DOMAIN("test")
    JST_BLOCK_DESCRIPTION("Environment Gate Test",
                          "Waits for an environment value.",
                          "A test-only block that stays incomplete until an environment value appears.")

    Result serialize(Parser::Map&) const override {
        return Result::SUCCESS;
    }

    Result deserialize(const Parser::Map&) override {
        return Result::SUCCESS;
    }

    std::size_t hash() const override {
        return 0;
    }
};

struct EnvironmentGateTestBlock : Block::Impl,
                                  DynamicConfig<EnvironmentGateTestBlockConfig> {
    Result define() override {
        return defineInterfaceOutput("out", "Output", "Output tensor.");
    }

    Result create() override {
        environmentGateTestState().createCalls += 1;

        if (!environment()->has("gate")) {
            JST_ERROR("[ENVIRONMENT_GATE_TEST] Waiting for 'gate' environment value.");
            return Result::INCOMPLETE;
        }

        const auto moduleConfig = std::make_shared<EnvironmentGateTestModuleConfig>();
        JST_CHECK(moduleCreate("source", moduleConfig, {}));
        JST_CHECK(moduleExposeOutput("out", {"source", "out"}));
        return Result::SUCCESS;
    }
};

struct EnvironmentGateSinkTestBlockConfig : Block::Config {
    JST_BLOCK_TYPE(environment_gate_sink_test)
    JST_BLOCK_DOMAIN("test")
    JST_BLOCK_DESCRIPTION("Environment Gate Sink Test",
                          "Consumes gated output.",
                          "A test-only sink block used to verify incomplete block retries.")

    Result serialize(Parser::Map&) const override {
        return Result::SUCCESS;
    }

    Result deserialize(const Parser::Map&) override {
        return Result::SUCCESS;
    }

    std::size_t hash() const override {
        return 0;
    }
};

struct EnvironmentGateSinkTestBlock : Block::Impl,
                                      DynamicConfig<EnvironmentGateSinkTestBlockConfig> {
    Result define() override {
        environmentGateTestState().sinkDefineCalls += 1;
        return defineInterfaceInput("in", "Input", "Input tensor.");
    }

    Result create() override {
        const auto moduleConfig = std::make_shared<RuntimeErrorSinkTestModuleConfig>();
        JST_CHECK(moduleCreate("sink", moduleConfig, {{"in", inputs().at("in")}}));
        return Result::SUCCESS;
    }
};

JST_REGISTER_MODULE(RuntimeErrorTestModule, DeviceType::CPU, RuntimeType::NATIVE, "generic");
JST_REGISTER_MODULE(RuntimeErrorSinkTestModule, DeviceType::CPU, RuntimeType::NATIVE, "generic");
JST_REGISTER_MODULE(EnvironmentGateTestModule, DeviceType::CPU, RuntimeType::NATIVE, "generic");
JST_REGISTER_BLOCK(RuntimeErrorTestBlock, {"runtime_error_test_module"});
JST_REGISTER_BLOCK(RuntimeErrorSinkTestBlock, {"runtime_error_sink_test_module"});
JST_REGISTER_BLOCK(EnvironmentGateTestBlock, {"environment_gate_test_module"});
JST_REGISTER_BLOCK(EnvironmentGateSinkTestBlock, {"runtime_error_sink_test_module"});

}  // namespace

using namespace Jetstream;
using namespace TestFlowgraph;

TEST_CASE_METHOD(FlowgraphFixture, "Block creation and destruction", "[flowgraph]") {
    SECTION("create single block") {
        auto result = flowgraph->blockCreate("gen1", "signal_generator", {}, {});
        REQUIRE(result == Result::SUCCESS);
        REQUIRE(flowgraph->view().size() == 1);
        REQUIRE(flowgraph->view().has("gen1"));
        REQUIRE(viewBlock("gen1").state == Block::State::Created);
    }

    SECTION("create multiple blocks") {
        REQUIRE(flowgraph->blockCreate("gen1", "signal_generator", {}, {}) == Result::SUCCESS);
        REQUIRE(flowgraph->blockCreate("gen2", "signal_generator", {}, {}) == Result::SUCCESS);
        REQUIRE(flowgraph->view().size() == 2);
        REQUIRE(viewBlock("gen1").state == Block::State::Created);
        REQUIRE(viewBlock("gen2").state == Block::State::Created);
    }

    SECTION("destroy block") {
        REQUIRE(flowgraph->blockCreate("gen1", "signal_generator", {}, {}) == Result::SUCCESS);
        REQUIRE(flowgraph->blockDestroy("gen1") == Result::SUCCESS);
        REQUIRE(flowgraph->view().empty());
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

TEST_CASE_METHOD(FlowgraphFixture, "Runtime module failure marks block errored", "[flowgraph][runtime]") {
    auto& state = runtimeErrorTestState();
    state.reset();

    REQUIRE(flowgraph->blockCreate("fail", "runtime_error_test", {}, {}) == Result::SUCCESS);
    TensorMap sinkInputs;
    sinkInputs["in"].requested("fail", "out");
    REQUIRE(flowgraph->blockCreate("sink", "runtime_error_sink_test", {}, sinkInputs) == Result::SUCCESS);
    REQUIRE(viewBlock("fail").state == Block::State::Created);
    REQUIRE(viewBlock("sink").state == Block::State::Created);

    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    const auto errored = viewBlock("fail");
    REQUIRE(errored.state == Block::State::Errored);
    REQUIRE(errored.diagnostic.find("Forced runtime failure") != std::string::npos);
    REQUIRE(errored.outputs.empty());
    REQUIRE(viewBlock("sink").state == Block::State::Incomplete);
    REQUIRE(state.calls == 1);
    REQUIRE(state.sinkCalls == 0);

    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    REQUIRE(state.calls == 1);
    REQUIRE(state.sinkCalls == 0);
}

TEST_CASE_METHOD(FlowgraphFixture, "Environment change retries incomplete blocks", "[flowgraph][environment]") {
    auto& state = environmentGateTestState();
    state.reset();

    REQUIRE(flowgraph->blockCreate("gated", "environment_gate_test", {}, {}) == Result::SUCCESS);
    REQUIRE(viewBlock("gated").state == Block::State::Incomplete);
    REQUIRE(state.createCalls == 1);

    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    REQUIRE(viewBlock("gated").state == Block::State::Incomplete);
    REQUIRE(state.createCalls == 1);

    SimpleMetaFixture gate;
    gate.order = 1;
    gate.label = "ready";
    REQUIRE(flowgraph->environment().set("gate", gate) == Result::SUCCESS);

    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    REQUIRE(viewBlock("gated").state == Block::State::Created);
    REQUIRE(state.createCalls == 2);

    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    REQUIRE(state.createCalls == 2);
}

TEST_CASE_METHOD(FlowgraphFixture, "Environment change retries incomplete downstream chain", "[flowgraph][environment]") {
    auto& state = environmentGateTestState();
    state.reset();

    REQUIRE(flowgraph->blockCreate("gated", "environment_gate_test", {}, {}) == Result::SUCCESS);
    TensorMap sinkInputs;
    sinkInputs["in"].requested("gated", "out");
    REQUIRE(flowgraph->blockCreate("sink", "environment_gate_sink_test", {}, sinkInputs) == Result::SUCCESS);
    REQUIRE(viewBlock("gated").state == Block::State::Incomplete);
    REQUIRE(viewBlock("sink").state == Block::State::Incomplete);

    SimpleMetaFixture gate;
    gate.order = 1;
    gate.label = "ready";
    REQUIRE(flowgraph->environment().set("gate", gate) == Result::SUCCESS);

    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    REQUIRE(viewBlock("gated").state == Block::State::Created);
    REQUIRE(viewBlock("sink").state == Block::State::Created);
}

TEST_CASE_METHOD(FlowgraphFixture, "Environment change skips unconnected incomplete blocks", "[flowgraph][environment]") {
    auto& state = environmentGateTestState();
    state.reset();

    REQUIRE(flowgraph->blockCreate("sink", "environment_gate_sink_test", {}, {}) == Result::SUCCESS);
    REQUIRE(viewBlock("sink").state == Block::State::Incomplete);
    REQUIRE(state.sinkDefineCalls == 1);

    SimpleMetaFixture gate;
    gate.order = 1;
    gate.label = "ready";
    REQUIRE(flowgraph->environment().set("gate", gate) == Result::SUCCESS);

    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    REQUIRE(viewBlock("sink").state == Block::State::Incomplete);
    REQUIRE(state.sinkDefineCalls == 1);
}

TEST_CASE_METHOD(FlowgraphFixture, "Environment value update does not retry incomplete blocks", "[flowgraph][environment]") {
    auto& state = environmentGateTestState();
    state.reset();

    REQUIRE(flowgraph->blockCreate("gated", "environment_gate_test", {}, {}) == Result::SUCCESS);
    REQUIRE(viewBlock("gated").state == Block::State::Incomplete);
    REQUIRE(state.createCalls == 1);

    SimpleMetaFixture telemetry;
    telemetry.order = 1;
    telemetry.label = "first";
    REQUIRE(flowgraph->environment().set("telemetry", telemetry) == Result::SUCCESS);

    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    REQUIRE(viewBlock("gated").state == Block::State::Incomplete);
    REQUIRE(state.createCalls == 2);

    telemetry.order = 2;
    telemetry.label = "second";
    REQUIRE(flowgraph->environment().set("telemetry", telemetry) == Result::SUCCESS);

    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    REQUIRE(viewBlock("gated").state == Block::State::Incomplete);
    REQUIRE(state.createCalls == 2);
}

TEST_CASE_METHOD(FlowgraphFixture, "Block connection", "[flowgraph]") {
    REQUIRE(flowgraph->blockCreate("gen1", "signal_generator", {}, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->blockCreate("gen2", "signal_generator", {}, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->blockCreate("gen3", "signal_generator", {}, {}) == Result::SUCCESS);

    TensorMap addInputs;
    addInputs["a"].requested("gen1", "signal");
    addInputs["b"].requested("gen2", "signal");
    REQUIRE(flowgraph->blockCreate("add1", "add", {}, addInputs) == Result::SUCCESS);
    REQUIRE(viewBlock("add1").state == Block::State::Created);

    SECTION("disconnect blocks") {
        auto result = flowgraph->blockDisconnect("add1", "a");
        REQUIRE((result == Result::SUCCESS || result == Result::INCOMPLETE));
        REQUIRE(viewBlock("add1").state == Block::State::Incomplete);
    }

    SECTION("reconnect after disconnect") {
        flowgraph->blockDisconnect("add1", "a");
        REQUIRE(viewBlock("add1").state == Block::State::Incomplete);

        auto result = flowgraph->blockConnect("add1", "a", "gen1", "signal");
        REQUIRE(result == Result::SUCCESS);
        REQUIRE(viewBlock("add1").state == Block::State::Created);
    }

    SECTION("reconnect to different source") {
        flowgraph->blockDisconnect("add1", "a");
        REQUIRE(viewBlock("add1").state == Block::State::Incomplete);

        auto result = flowgraph->blockConnect("add1", "a", "gen3", "signal");
        REQUIRE(result == Result::SUCCESS);
        REQUIRE(viewBlock("add1").state == Block::State::Created);
    }

    SECTION("connect to nonexistent block fails") {
        auto result = flowgraph->blockConnect("add1", "a", "nonexistent", "signal");
        REQUIRE(result == Result::ERROR);
    }

    SECTION("connect to nonexistent source port leaves block incomplete") {
        flowgraph->blockDisconnect("add1", "a");
        auto result = flowgraph->blockConnect("add1", "a", "gen1", "nonexistent");
        REQUIRE((result == Result::SUCCESS || result == Result::INCOMPLETE));
        REQUIRE(viewBlock("add1").state == Block::State::Incomplete);
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
    REQUIRE(flowgraph->view().size() == 4);
    REQUIRE(viewBlock("gen1").state == Block::State::Created);
    REQUIRE(viewBlock("gen2").state == Block::State::Created);
    REQUIRE(viewBlock("add1").state == Block::State::Created);
    REQUIRE(viewBlock("add2").state == Block::State::Created);

    SECTION("disconnecting upstream marks downstream incomplete") {
        auto result = flowgraph->blockDisconnect("add1", "a");
        REQUIRE((result == Result::SUCCESS || result == Result::INCOMPLETE));

        // add1 should be incomplete, add2 should be incomplete (unresolved input)
        REQUIRE(viewBlock("add1").state == Block::State::Incomplete);
        REQUIRE(viewBlock("add2").state == Block::State::Incomplete);
    }

    SECTION("reconnecting upstream restores downstream") {
        flowgraph->blockDisconnect("add1", "a");
        REQUIRE(viewBlock("add1").state == Block::State::Incomplete);
        REQUIRE(viewBlock("add2").state == Block::State::Incomplete);

        auto result = flowgraph->blockConnect("add1", "a", "gen1", "signal");
        REQUIRE(result == Result::SUCCESS);
        REQUIRE(viewBlock("add1").state == Block::State::Created);
        REQUIRE(viewBlock("add2").state == Block::State::Created);
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

    REQUIRE(viewBlock("add1").state == Block::State::Created);
    REQUIRE(viewBlock("add2").state == Block::State::Created);

    SECTION("destroying upstream marks downstream incomplete") {
        REQUIRE(flowgraph->blockDestroy("gen1") == Result::SUCCESS);

        REQUIRE(flowgraph->view().has("add1"));
        REQUIRE(viewBlock("add1").state == Block::State::Incomplete);
        REQUIRE(viewBlock("add2").state == Block::State::Incomplete);
    }

    SECTION("destroying middle block marks downstream incomplete") {
        REQUIRE(flowgraph->blockDestroy("add1") == Result::SUCCESS);

        REQUIRE_FALSE(flowgraph->view().has("add1"));
        REQUIRE(flowgraph->view().has("add2"));
        REQUIRE(viewBlock("add2").state == Block::State::Incomplete);
    }

    SECTION("destroying all upstream sources marks downstream incomplete") {
        REQUIRE(flowgraph->blockDestroy("gen1") == Result::SUCCESS);
        REQUIRE(flowgraph->blockDestroy("gen2") == Result::SUCCESS);

        REQUIRE(viewBlock("add1").state == Block::State::Incomplete);
        REQUIRE(viewBlock("add2").state == Block::State::Incomplete);
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

    REQUIRE(flowgraph->view().size() == 5);
    REQUIRE(viewBlock("add0").state == Block::State::Created);
    REQUIRE(viewBlock("add1").state == Block::State::Created);
    REQUIRE(viewBlock("add2").state == Block::State::Created);

    SECTION("destroying middle block preserves other connections in downstream") {
        REQUIRE(flowgraph->blockDestroy("add0") == Result::SUCCESS);

        // add0 should be gone
        REQUIRE_FALSE(flowgraph->view().has("add0"));

        // add1 should exist but be incomplete (lost a from add0)
        REQUIRE(flowgraph->view().has("add1"));
        REQUIRE(viewBlock("add1").state == Block::State::Incomplete);

        // add2 should exist but be incomplete (add1 is incomplete so no valid output)
        REQUIRE(flowgraph->view().has("add2"));
        REQUIRE(viewBlock("add2").state == Block::State::Incomplete);

        // gen1 and gen2 should still be created
        REQUIRE(viewBlock("gen1").state == Block::State::Created);
        REQUIRE(viewBlock("gen2").state == Block::State::Created);
    }

    SECTION("reconnecting after middle block destruction restores chain") {
        REQUIRE(flowgraph->blockDestroy("add0") == Result::SUCCESS);

        // Reconnect add1.a to gen1 directly
        REQUIRE(flowgraph->blockConnect("add1", "a", "gen1", "signal") == Result::SUCCESS);

        // Both add1 and add2 should now be Created
        REQUIRE(viewBlock("add1").state == Block::State::Created);
        REQUIRE(viewBlock("add2").state == Block::State::Created);
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

    REQUIRE(flowgraph->view().size() == 6);

    SECTION("destroying early block marks entire downstream chain incomplete") {
        REQUIRE(flowgraph->blockDestroy("add1") == Result::SUCCESS);

        REQUIRE_FALSE(flowgraph->view().has("add1"));
        REQUIRE(viewBlock("add2").state == Block::State::Incomplete);
        REQUIRE(viewBlock("add3").state == Block::State::Incomplete);
        REQUIRE(viewBlock("add4").state == Block::State::Incomplete);
    }

    SECTION("destroying middle of chain marks only downstream incomplete") {
        REQUIRE(flowgraph->blockDestroy("add2") == Result::SUCCESS);

        REQUIRE(viewBlock("add1").state == Block::State::Created);
        REQUIRE_FALSE(flowgraph->view().has("add2"));
        REQUIRE(viewBlock("add3").state == Block::State::Incomplete);
        REQUIRE(viewBlock("add4").state == Block::State::Incomplete);
    }

    SECTION("destroying end of chain leaves upstream created") {
        REQUIRE(flowgraph->blockDestroy("add4") == Result::SUCCESS);

        REQUIRE(viewBlock("add1").state == Block::State::Created);
        REQUIRE(viewBlock("add2").state == Block::State::Created);
        REQUIRE(viewBlock("add3").state == Block::State::Created);
        REQUIRE_FALSE(flowgraph->view().has("add4"));
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

    REQUIRE(flowgraph->view().size() == 5);
    REQUIRE(viewBlock("add3").state == Block::State::Created);

    SECTION("destroying one branch leaves add3 incomplete but preserves other branch") {
        REQUIRE(flowgraph->blockDestroy("add1") == Result::SUCCESS);

        REQUIRE_FALSE(flowgraph->view().has("add1"));
        REQUIRE(viewBlock("add2").state == Block::State::Created);
        REQUIRE(viewBlock("add3").state == Block::State::Incomplete);
    }

    SECTION("destroying other branch leaves add3 incomplete but preserves first branch") {
        REQUIRE(flowgraph->blockDestroy("add2") == Result::SUCCESS);

        REQUIRE(viewBlock("add1").state == Block::State::Created);
        REQUIRE_FALSE(flowgraph->view().has("add2"));
        REQUIRE(viewBlock("add3").state == Block::State::Incomplete);
    }

    SECTION("destroying source marks all downstream incomplete") {
        REQUIRE(flowgraph->blockDestroy("gen1") == Result::SUCCESS);

        REQUIRE_FALSE(flowgraph->view().has("gen1"));
        REQUIRE(viewBlock("add1").state == Block::State::Incomplete);
        REQUIRE(viewBlock("add2").state == Block::State::Incomplete);
        REQUIRE(viewBlock("add3").state == Block::State::Incomplete);
    }

    SECTION("reconnecting both branches restores add3") {
        REQUIRE(flowgraph->blockDestroy("add1") == Result::SUCCESS);
        REQUIRE(viewBlock("add3").state == Block::State::Incomplete);

        // Reconnect add3.a to gen1 directly (bypassing removed add1)
        REQUIRE(flowgraph->blockConnect("add3", "a", "gen1", "signal") == Result::SUCCESS);
        REQUIRE(viewBlock("add3").state == Block::State::Created);
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

    REQUIRE(flowgraph->view().size() == 6);

    SECTION("destroying fan-out source marks all dependents incomplete") {
        REQUIRE(flowgraph->blockDestroy("add0") == Result::SUCCESS);

        REQUIRE_FALSE(flowgraph->view().has("add0"));
        REQUIRE(viewBlock("add1").state == Block::State::Incomplete);
        REQUIRE(viewBlock("add2").state == Block::State::Incomplete);
        REQUIRE(viewBlock("add3").state == Block::State::Incomplete);

        // gen1 and gen2 should still be created
        REQUIRE(viewBlock("gen1").state == Block::State::Created);
        REQUIRE(viewBlock("gen2").state == Block::State::Created);
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

        REQUIRE_FALSE(flowgraph->view().has("add1"));
        // add2 still exists (not recreated, may have invalid state - internal use)
        REQUIRE(flowgraph->view().has("add2"));
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
        REQUIRE_FALSE(flowgraph->view().has("add1"));
        REQUIRE(viewBlock("add2").state == Block::State::Incomplete);

        // Recreate add1
        REQUIRE(flowgraph->blockCreate("add1", "add", {}, add1Inputs) == Result::SUCCESS);
        REQUIRE(viewBlock("add1").state == Block::State::Created);

        // Reconnect add2 to the new add1
        REQUIRE(flowgraph->blockConnect("add2", "a", "add1", "sum") == Result::SUCCESS);
        REQUIRE(viewBlock("add2").state == Block::State::Created);
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

    REQUIRE(viewBlock("add1").state == Block::State::Created);

    SECTION("destroying source disconnects all inputs from that source") {
        REQUIRE(flowgraph->blockDestroy("gen1") == Result::SUCCESS);

        REQUIRE_FALSE(flowgraph->view().has("gen1"));
        REQUIRE(flowgraph->view().has("add1"));
        REQUIRE(viewBlock("add1").state == Block::State::Incomplete);
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

    REQUIRE(viewBlock("add1").state == Block::State::Created);
    REQUIRE(viewBlock("add2").state == Block::State::Created);

    SECTION("destroying shared source marks all downstream incomplete") {
        REQUIRE(flowgraph->blockDestroy("gen1") == Result::SUCCESS);

        REQUIRE(viewBlock("add1").state == Block::State::Incomplete);
        REQUIRE(viewBlock("add2").state == Block::State::Incomplete);
    }

    SECTION("disconnecting from shared source marks only that block incomplete") {
        flowgraph->blockDisconnect("add1", "a");

        REQUIRE(viewBlock("add1").state == Block::State::Incomplete);
        REQUIRE(viewBlock("add2").state == Block::State::Created);
    }
}

TEST_CASE_METHOD(FlowgraphFixture, "Block reconfiguration", "[flowgraph]") {
    REQUIRE(flowgraph->blockCreate("gen1", "signal_generator", {}, {}) == Result::SUCCESS);
    REQUIRE(viewBlock("gen1").state == Block::State::Created);

    SECTION("reconfigure existing block") {
        Parser::Map newConfig;
        newConfig["bufferSize"] = std::string("2048");
        auto result = flowgraph->blockReconfigure("gen1", newConfig);
        REQUIRE(result == Result::SUCCESS);
        REQUIRE(viewBlock("gen1").state == Block::State::Created);
    }

    SECTION("reconfigure nonexistent block fails") {
        Parser::Map newConfig;
        auto result = flowgraph->blockReconfigure("nonexistent", newConfig);
        REQUIRE(result == Result::ERROR);
    }
}

TEST_CASE_METHOD(FlowgraphFixture, "Block recreation", "[flowgraph][recreation]") {
    REQUIRE(flowgraph->blockCreate("gen1", "signal_generator", {}, {}) == Result::SUCCESS);
    REQUIRE(viewBlock("gen1").state == Block::State::Created);

    SECTION("recreate single block") {
        Parser::Map newConfig;
        newConfig["bufferSize"] = std::string("4096");
        auto result = flowgraph->blockRecreate("gen1", newConfig);
        REQUIRE(result == Result::SUCCESS);
        REQUIRE(flowgraph->view().size() == 1);
        REQUIRE(flowgraph->view().has("gen1"));
        REQUIRE(viewBlock("gen1").state == Block::State::Created);
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

    const auto block = viewBlock("gen1");
    const auto implementations = Registry::ListAvailableModules("signal_generator",
                                                                 block.device,
                                                                 block.runtime,
                                                                 kSignalGeneratorTestProvider);

    REQUIRE(implementations.size() == 1);
    REQUIRE(block.provider != implementations.front().provider);

    REQUIRE(flowgraph->blockRecreate("gen1",
                                     config,
                                     implementations.front().device,
                                     implementations.front().runtime,
                                     implementations.front().provider) == Result::SUCCESS);

    const auto recreated = viewBlock("gen1");
    REQUIRE(recreated.state == Block::State::Created);
    REQUIRE(recreated.device == implementations.front().device);
    REQUIRE(recreated.runtime == implementations.front().runtime);
    REQUIRE(recreated.provider == implementations.front().provider);
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

    REQUIRE(flowgraph->view().size() == 5);
    REQUIRE(viewBlock("add1").state == Block::State::Created);
    REQUIRE(viewBlock("add2").state == Block::State::Created);
    REQUIRE(viewBlock("add3").state == Block::State::Created);

    SECTION("recreating source recreates entire downstream chain") {
        Parser::Map newConfig;
        auto result = flowgraph->blockRecreate("gen1", newConfig);
        REQUIRE(result == Result::SUCCESS);

        // All blocks should still exist and be created
        REQUIRE(flowgraph->view().size() == 5);
        REQUIRE(viewBlock("gen1").state == Block::State::Created);
        REQUIRE(viewBlock("gen2").state == Block::State::Created);
        REQUIRE(viewBlock("add1").state == Block::State::Created);
        REQUIRE(viewBlock("add2").state == Block::State::Created);
        REQUIRE(viewBlock("add3").state == Block::State::Created);
    }

    SECTION("recreating middle block recreates downstream only") {
        Parser::Map newConfig;
        auto result = flowgraph->blockRecreate("add1", newConfig);
        REQUIRE(result == Result::SUCCESS);

        // All blocks should still exist and be created
        REQUIRE(flowgraph->view().size() == 5);
        REQUIRE(viewBlock("gen1").state == Block::State::Created);
        REQUIRE(viewBlock("gen2").state == Block::State::Created);
        REQUIRE(viewBlock("add1").state == Block::State::Created);
        REQUIRE(viewBlock("add2").state == Block::State::Created);
        REQUIRE(viewBlock("add3").state == Block::State::Created);
    }

    SECTION("recreating end block recreates only that block") {
        Parser::Map newConfig;
        auto result = flowgraph->blockRecreate("add3", newConfig);
        REQUIRE(result == Result::SUCCESS);

        // All blocks should still exist and be created
        REQUIRE(flowgraph->view().size() == 5);
        REQUIRE(viewBlock("add1").state == Block::State::Created);
        REQUIRE(viewBlock("add2").state == Block::State::Created);
        REQUIRE(viewBlock("add3").state == Block::State::Created);
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

    REQUIRE(flowgraph->view().size() == 5);
    REQUIRE(viewBlock("add3").state == Block::State::Created);

    SECTION("recreating source recreates all downstream including diamond") {
        Parser::Map newConfig;
        auto result = flowgraph->blockRecreate("gen1", newConfig);
        REQUIRE(result == Result::SUCCESS);

        REQUIRE(flowgraph->view().size() == 5);
        REQUIRE(viewBlock("gen1").state == Block::State::Created);
        REQUIRE(viewBlock("add1").state == Block::State::Created);
        REQUIRE(viewBlock("add2").state == Block::State::Created);
        REQUIRE(viewBlock("add3").state == Block::State::Created);
    }

    SECTION("recreating one branch recreates that branch and convergence point") {
        Parser::Map newConfig;
        auto result = flowgraph->blockRecreate("add1", newConfig);
        REQUIRE(result == Result::SUCCESS);

        REQUIRE(flowgraph->view().size() == 5);
        REQUIRE(viewBlock("add1").state == Block::State::Created);
        REQUIRE(viewBlock("add2").state == Block::State::Created);
        REQUIRE(viewBlock("add3").state == Block::State::Created);
    }
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Block recreation topologically orders uneven converging branches",
                 "[flowgraph][recreation]") {
    // Topology:
    //        ┌──► short ───────────────────┐
    // gen1 ──┤                             ├──► merge
    //        └──► long1 ──► long2 ──► long3 ──┘

    REQUIRE(flowgraph->blockCreate("gen1", "signal_generator", {}, {}) == Result::SUCCESS);

    TensorMap shortInputs;
    shortInputs["buffer"].requested("gen1", "signal");
    REQUIRE(flowgraph->blockCreate("short", "duplicate", {}, shortInputs) == Result::SUCCESS);

    TensorMap long1Inputs;
    long1Inputs["buffer"].requested("gen1", "signal");
    REQUIRE(flowgraph->blockCreate("long1", "duplicate", {}, long1Inputs) == Result::SUCCESS);

    TensorMap long2Inputs;
    long2Inputs["buffer"].requested("long1", "buffer");
    REQUIRE(flowgraph->blockCreate("long2", "duplicate", {}, long2Inputs) == Result::SUCCESS);

    TensorMap long3Inputs;
    long3Inputs["buffer"].requested("long2", "buffer");
    REQUIRE(flowgraph->blockCreate("long3", "duplicate", {}, long3Inputs) == Result::SUCCESS);

    TensorMap mergeInputs;
    mergeInputs["a"].requested("short", "buffer");
    mergeInputs["b"].requested("long3", "buffer");
    REQUIRE(flowgraph->blockCreate("merge", "add", {}, mergeInputs) == Result::SUCCESS);

    Parser::Map newConfig;
    newConfig["bufferSize"] = std::string("1024");
    REQUIRE(flowgraph->blockReconfigure("gen1", newConfig) == Result::SUCCESS);

    for (const auto& name : {"gen1", "short", "long1", "long2", "long3", "merge"}) {
        REQUIRE(viewBlock(name).state == Block::State::Created);
    }

    const auto merge = viewBlock("merge");
    REQUIRE(merge.outputs.at("sum").tensor.shape() == Shape{1024});
    REQUIRE(merge.inputs.at("b").external.has_value());
    REQUIRE(merge.inputs.at("b").external->block == "long3");
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

    REQUIRE(flowgraph->view().size() == 6);

    SECTION("recreating fan-out source recreates all dependents") {
        Parser::Map newConfig;
        auto result = flowgraph->blockRecreate("add0", newConfig);
        REQUIRE(result == Result::SUCCESS);

        REQUIRE(flowgraph->view().size() == 6);
        REQUIRE(viewBlock("add0").state == Block::State::Created);
        REQUIRE(viewBlock("add1").state == Block::State::Created);
        REQUIRE(viewBlock("add2").state == Block::State::Created);
        REQUIRE(viewBlock("add3").state == Block::State::Created);
    }

    SECTION("recreating one fan-out target does not affect siblings") {
        Parser::Map newConfig;
        auto result = flowgraph->blockRecreate("add1", newConfig);
        REQUIRE(result == Result::SUCCESS);

        REQUIRE(flowgraph->view().size() == 6);
        REQUIRE(viewBlock("add0").state == Block::State::Created);
        REQUIRE(viewBlock("add1").state == Block::State::Created);
        REQUIRE(viewBlock("add2").state == Block::State::Created);
        REQUIRE(viewBlock("add3").state == Block::State::Created);
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
        REQUIRE(viewBlock("add1").state == Block::State::Created);
        REQUIRE(viewBlock("add2").state == Block::State::Created);

        // Verify add2 still has its connection to add1
        const auto add2Ptr = viewBlock("add2");
        const auto& inputs = add2Ptr.inputs;
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

    REQUIRE(viewBlock("gen1").state == Block::State::Created);
    REQUIRE(viewBlock("gen2").state == Block::State::Created);
    REQUIRE(viewBlock("add1").state == Block::State::Created);

    flowgraph->setTitle("Test Flowgraph");
    flowgraph->setAuthor("Test Author");

    SECTION("export to blob") {
        std::vector<char> blob;
        auto result = flowgraph->exportToBlob(blob);
        REQUIRE(result == Result::SUCCESS);
        REQUIRE(!blob.empty());

        const std::string yaml(blob.begin(), blob.end());
        REQUIRE(yaml.starts_with("---\nversion: 2\n"));
    }

    SECTION("export and reimport") {
        std::vector<char> blob;
        REQUIRE(flowgraph->exportToBlob(blob) == Result::SUCCESS);

        // Destroy current blocks
        std::vector<std::string> names;
        REQUIRE(flowgraph->view().keys(names) == Result::SUCCESS);
        for (const auto& name : names) {
            flowgraph->blockDestroy(name, false);
        }
        REQUIRE(flowgraph->view().empty());

        // Reimport
        REQUIRE(flowgraph->importFromBlob(blob) == Result::SUCCESS);
        REQUIRE(flowgraph->view().size() == 3);
        REQUIRE(viewBlock("gen1").state == Block::State::Created);
        REQUIRE(viewBlock("gen2").state == Block::State::Created);
        REQUIRE(viewBlock("add1").state == Block::State::Created);
        REQUIRE(flowgraph->title() == "Test Flowgraph");
        REQUIRE(flowgraph->author() == "Test Author");
    }
}
