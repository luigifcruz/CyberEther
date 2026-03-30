#include <catch2/catch_test_macros.hpp>

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "jetstream/detail/module_impl.hh"
#include "jetstream/registry.hh"
#include "jetstream/runtime_context_native_cpu.hh"
#include "jetstream/scheduler.hh"
#include "jetstream/scheduler_context.hh"

namespace {

using namespace Jetstream;

struct SkipTestState {
    std::unordered_map<std::string, Result> results;
    std::unordered_map<std::string, U64> calls;

    void reset() {
        results.clear();
        calls.clear();
    }

    void setResult(const std::string& module, const Result result) {
        results[module] = result;
    }

    Result resultFor(const std::string& module) const {
        if (!results.contains(module)) {
            return Result::SUCCESS;
        }

        return results.at(module);
    }

    void noteCall(const std::string& module) {
        calls[module] += 1;
    }

    U64 callCount(const std::string& module) const {
        if (!calls.contains(module)) {
            return 0;
        }

        return calls.at(module);
    }
};

SkipTestState& skipTestState() {
    static SkipTestState state;
    return state;
}

struct SkipTestSourceConfig : Module::Config {
    JST_MODULE_TYPE(skip_test_source)

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

struct SkipTestPassthroughConfig : Module::Config {
    JST_MODULE_TYPE(skip_test_passthrough)

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

struct SkipTestMergeConfig : Module::Config {
    JST_MODULE_TYPE(skip_test_merge)

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

struct SkipTestSourceModule : Module::Impl,
                              DynamicConfig<SkipTestSourceConfig>,
                              NativeCpuRuntimeContext,
                              Scheduler::Context {
    Result define() override {
        JST_CHECK(defineInterfaceOutput("out"));
        return Result::SUCCESS;
    }

    Result create() override {
        JST_CHECK(output.create(DeviceType::CPU, DataType::F32, {1}));
        output.at<F32>(0) = 1.0f;
        outputs()["out"].produced(name(), "out", output);
        return Result::SUCCESS;
    }

    Result computeSubmit() override {
        auto& state = skipTestState();
        state.noteCall(name());
        return state.resultFor(name());
    }

    Tensor output;
};

struct SkipTestPassthroughModule : Module::Impl,
                                   DynamicConfig<SkipTestPassthroughConfig>,
                                   NativeCpuRuntimeContext,
                                   Scheduler::Context {
    Result define() override {
        JST_CHECK(defineInterfaceInput("in"));
        JST_CHECK(defineInterfaceOutput("out"));
        return Result::SUCCESS;
    }

    Result create() override {
        output = inputs().at("in").tensor.clone();
        outputs()["out"].produced(name(), "out", output);
        return Result::SUCCESS;
    }

    Result computeSubmit() override {
        auto& state = skipTestState();
        state.noteCall(name());
        return state.resultFor(name());
    }

    Tensor output;
};

struct SkipTestMergeModule : Module::Impl,
                             DynamicConfig<SkipTestMergeConfig>,
                             NativeCpuRuntimeContext,
                             Scheduler::Context {
    Result define() override {
        JST_CHECK(defineInterfaceInput("left"));
        JST_CHECK(defineInterfaceInput("right"));
        JST_CHECK(defineInterfaceOutput("out"));
        return Result::SUCCESS;
    }

    Result create() override {
        output = inputs().at("left").tensor.clone();
        outputs()["out"].produced(name(), "out", output);
        return Result::SUCCESS;
    }

    Result computeSubmit() override {
        auto& state = skipTestState();
        state.noteCall(name());
        return state.resultFor(name());
    }

    Tensor output;
};

JST_REGISTER_MODULE(SkipTestSourceModule, DeviceType::CPU, RuntimeType::NATIVE, "generic");
JST_REGISTER_MODULE(SkipTestPassthroughModule, DeviceType::CPU, RuntimeType::NATIVE, "generic");
JST_REGISTER_MODULE(SkipTestMergeModule, DeviceType::CPU, RuntimeType::NATIVE, "generic");

Tensor makeTensor() {
    Tensor tensor;

    if (tensor.create(DeviceType::CPU, DataType::F32, {1}) != Result::SUCCESS) {
        throw std::runtime_error("failed to create test tensor");
    }

    tensor.at<F32>(0) = 1.0f;
    return tensor;
}

std::shared_ptr<Module> createModule(const std::string& type,
                                     const std::string& name,
                                     const TensorMap& inputs = {}) {
    std::shared_ptr<Module> module;

    if (Registry::BuildModule(type, DeviceType::CPU, RuntimeType::NATIVE, "generic", module) != Result::SUCCESS) {
        throw std::runtime_error("failed to build test module: " + type);
    }

    Parser::Map config;

    if (module->create(name, config, inputs) != Result::SUCCESS) {
        throw std::runtime_error("failed to create test module: " + name);
    }

    return module;
}

void destroyModules(std::vector<std::shared_ptr<Module>>& modules) {
    for (auto it = modules.rbegin(); it != modules.rend(); ++it) {
        REQUIRE((*it)->destroy() == Result::SUCCESS);
    }
}

}  // namespace

TEST_CASE("Scheduler propagates SKIP to downstream modules", "[runtime][scheduler][skip]") {
    auto& state = skipTestState();
    state.reset();

    Scheduler scheduler(SchedulerType::SYNCHRONOUS);
    REQUIRE(scheduler.create(nullptr) == Result::SUCCESS);

    auto sourceSkip = createModule("skip_test_source", "source_skip");
    auto sourceRun = createModule("skip_test_source", "source_run");
    
    TensorMap downstreamSkipInputs;
    downstreamSkipInputs["in"].produced("source_skip", "out", makeTensor());
    auto downstreamSkip = createModule("skip_test_passthrough", "downstream_skip", downstreamSkipInputs);
    
    TensorMap downstreamRunInputs;
    downstreamRunInputs["in"].produced("source_run", "out", makeTensor());
    auto downstreamRun = createModule("skip_test_passthrough", "downstream_run", downstreamRunInputs);

    std::vector<std::shared_ptr<Module>> modules = {
        sourceSkip,
        sourceRun,
        downstreamSkip,
        downstreamRun,
    };

    for (const auto& module : modules) {
        REQUIRE(scheduler.add(module) == Result::SUCCESS);
    }

    state.setResult("source_skip", Result::SKIP);

    REQUIRE(scheduler.compute() == Result::SUCCESS);

    REQUIRE(state.callCount("source_skip") == 1);
    REQUIRE(state.callCount("source_run") == 1);
    REQUIRE(state.callCount("downstream_skip") == 0);
    REQUIRE(state.callCount("downstream_run") == 1);

    for (auto it = modules.rbegin(); it != modules.rend(); ++it) {
        REQUIRE(scheduler.remove(*it) == Result::SUCCESS);
    }

    REQUIRE(scheduler.destroy() == Result::SUCCESS);
    destroyModules(modules);
}

TEST_CASE("Runtime propagates SKIP across compute barriers", "[runtime][skip][barrier]") {
    auto& state = skipTestState();
    state.reset();

    auto source = createModule("skip_test_source", "barrier_source");
    
    TensorMap sinkInputs;
    sinkInputs["in"].produced("barrier_source", "out", makeTensor());
    auto sink = createModule("skip_test_passthrough", "barrier_sink", sinkInputs);

    Runtime upstream("upstream", DeviceType::CPU, RuntimeType::NATIVE);
    Runtime downstream("downstream", DeviceType::CPU, RuntimeType::NATIVE);

    REQUIRE(upstream.create({{"barrier_source", source}}) == Result::SUCCESS);
    REQUIRE(downstream.create({{"barrier_sink", sink}}) == Result::SUCCESS);

    state.setResult("barrier_source", Result::SKIP);

    std::unordered_set<std::string> skippedModules;
    REQUIRE(upstream.compute({"barrier_source"}, skippedModules) == Result::SUCCESS);
    REQUIRE(downstream.compute({"barrier_sink"}, skippedModules) == Result::SUCCESS);

    REQUIRE(state.callCount("barrier_source") == 1);
    REQUIRE(state.callCount("barrier_sink") == 0);

    REQUIRE(downstream.destroy() == Result::SUCCESS);
    REQUIRE(upstream.destroy() == Result::SUCCESS);
    REQUIRE(sink->destroy() == Result::SUCCESS);
    REQUIRE(source->destroy() == Result::SUCCESS);
}

TEST_CASE("Runtime skips multi-input consumers when any upstream module skips", "[runtime][skip][fanin]") {
    auto& state = skipTestState();
    state.reset();

    auto sourceSkip = createModule("skip_test_source", "fanin_source_skip");
    auto sourceRun = createModule("skip_test_source", "fanin_source_run");
    
    TensorMap mergeInputs;
    mergeInputs["left"].produced("fanin_source_skip", "out", makeTensor());
    mergeInputs["right"].produced("fanin_source_run", "out", makeTensor());
    auto merge = createModule("skip_test_merge", "fanin_merge", mergeInputs);

    Runtime runtime("fanin", DeviceType::CPU, RuntimeType::NATIVE);
    REQUIRE(runtime.create({
        {"fanin_source_skip", sourceSkip},
        {"fanin_source_run", sourceRun},
        {"fanin_merge", merge},
    }) == Result::SUCCESS);

    state.setResult("fanin_source_skip", Result::SKIP);

    std::unordered_set<std::string> skippedModules;
    REQUIRE(runtime.compute({
        "fanin_source_skip",
        "fanin_source_run",
        "fanin_merge",
    }, skippedModules) == Result::SUCCESS);

    REQUIRE(state.callCount("fanin_source_skip") == 1);
    REQUIRE(state.callCount("fanin_source_run") == 1);
    REQUIRE(state.callCount("fanin_merge") == 0);

    REQUIRE(runtime.destroy() == Result::SUCCESS);
    REQUIRE(merge->destroy() == Result::SUCCESS);
    REQUIRE(sourceRun->destroy() == Result::SUCCESS);
    REQUIRE(sourceSkip->destroy() == Result::SUCCESS);
}
