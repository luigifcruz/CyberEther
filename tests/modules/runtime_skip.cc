#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "jetstream/detail/module_impl.hh"
#include "jetstream/registry.hh"
#include "jetstream/runtime_context_native_cpu.hh"
#include "jetstream/runtime_context_python.hh"
#include "jetstream/scheduler.hh"
#include "jetstream/scheduler_context.hh"

namespace {

using namespace Jetstream;

struct SkipTestState {
    std::unordered_map<std::string, Result> results;
    std::unordered_map<std::string, std::vector<Result>> resultSequences;
    std::unordered_map<std::string, std::size_t> resultCursors;
    std::unordered_map<std::string, U64> calls;
    std::unordered_map<std::string, U64> pendingCalls;
    std::unordered_map<std::string, U64> presentCalls;
    std::unordered_map<std::string, U64> initializes;
    std::unordered_map<std::string, U64> deinitializes;
    std::unordered_map<std::string, Module::Taint> taints;
    std::unordered_map<std::string, std::chrono::milliseconds> delays;
    std::unordered_set<std::string> surfaces;
    U64 initializeCount = 0;
    U64 failInitializeAt = 0;

    void reset() {
        results.clear();
        resultSequences.clear();
        resultCursors.clear();
        calls.clear();
        pendingCalls.clear();
        presentCalls.clear();
        initializes.clear();
        deinitializes.clear();
        taints.clear();
        delays.clear();
        surfaces.clear();
        initializeCount = 0;
        failInitializeAt = 0;
    }

    void setResult(const std::string& module, const Result result) {
        results[module] = result;
    }

    void setResults(const std::string& module, std::vector<Result> sequence) {
        resultSequences[module] = std::move(sequence);
        resultCursors[module] = 0;
    }

    Result resultFor(const std::string& module) {
        if (resultSequences.contains(module)) {
            const auto& sequence = resultSequences.at(module);
            auto& cursor = resultCursors[module];
            if (cursor < sequence.size()) {
                return sequence[cursor++];
            }
        }

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

    void notePending(const std::string& module) {
        pendingCalls[module] += 1;
    }

    U64 pendingCallCount(const std::string& module) const {
        const auto it = pendingCalls.find(module);
        return it == pendingCalls.end() ? 0 : it->second;
    }

    void notePresent(const std::string& module) {
        presentCalls[module] += 1;
    }

    U64 presentCallCount(const std::string& module) const {
        const auto it = presentCalls.find(module);
        return it == presentCalls.end() ? 0 : it->second;
    }

    void setTaint(const std::string& module, Module::Taint taint) {
        taints[module] = taint;
    }

    Module::Taint taintFor(const std::string& module) const {
        const auto it = taints.find(module);
        return it == taints.end() ? Module::Taint::CLEAN : it->second;
    }

    void setDelay(const std::string& module, std::chrono::milliseconds delay) {
        delays[module] = delay;
    }

    std::chrono::milliseconds delayFor(const std::string& module) const {
        const auto it = delays.find(module);
        return it == delays.end() ? std::chrono::milliseconds{0} : it->second;
    }

    void setSurface(const std::string& module) {
        surfaces.insert(module);
    }

    bool isSurface(const std::string& module) const {
        return surfaces.contains(module);
    }

    void setInitializeFailureAt(const U64 count) {
        failInitializeAt = count;
    }

    Result noteInitialize(const std::string& module) {
        initializes[module] += 1;
        initializeCount += 1;

        if (failInitializeAt != 0 && initializeCount >= failInitializeAt) {
            return Result::ERROR;
        }

        return Result::SUCCESS;
    }

    void noteDeinitialize(const std::string& module) {
        deinitializes[module] += 1;
    }

    U64 initializeCallCount(const std::string& module) const {
        if (!initializes.contains(module)) {
            return 0;
        }

        return initializes.at(module);
    }

    U64 deinitializeCallCount(const std::string& module) const {
        if (!deinitializes.contains(module)) {
            return 0;
        }

        return deinitializes.at(module);
    }

    U64 totalDeinitializeCallCount() const {
        U64 total = 0;
        for (const auto& [_, count] : deinitializes) {
            total += count;
        }

        return total;
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

struct SkipTestFailInitConfig : Module::Config {
    JST_MODULE_TYPE(skip_test_fail_init)

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

struct SkipTestPythonConfig : Module::Config {
    JST_MODULE_TYPE(skip_test_python)

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
        auto& state = skipTestState();
        if (state.isSurface(name())) {
            JST_CHECK(defineTaint(Module::Taint::SURFACE));
        }
        if (state.taintFor(name()) != Module::Taint::CLEAN) {
            JST_CHECK(defineTaint(state.taintFor(name())));
        }

        JST_CHECK(defineInterfaceOutput("out"));
        return Result::SUCCESS;
    }

    Result create() override {
        JST_CHECK(output.create(DeviceType::CPU, DataType::F32, {1}));
        output.at<F32>(0) = 1.0f;
        outputs()["out"].produced(name(), "out", output);
        return Result::SUCCESS;
    }

    Result computeInitialize() override {
        const auto result = skipTestState().noteInitialize(name());
        if (result != Result::SUCCESS) {
            JST_ERROR("[SKIP_TEST_SOURCE] Forced runtime initialization failure.");
        }

        return result;
    }

    Result computeDeinitialize() override {
        skipTestState().noteDeinitialize(name());
        return Result::SUCCESS;
    }

    Result computeSubmit() override {
        auto& state = skipTestState();
        if (state.delayFor(name()).count() > 0) {
            std::this_thread::sleep_for(state.delayFor(name()));
        }
        state.noteCall(name());
        return state.resultFor(name());
    }

    Result presentSubmit() override {
        skipTestState().notePresent(name());
        return Result::SUCCESS;
    }

    Result hasPendingCompute() override {
        skipTestState().notePending(name());
        return Result::SUCCESS;
    }

    Tensor output;
};

struct SkipTestPassthroughModule : Module::Impl,
                                   DynamicConfig<SkipTestPassthroughConfig>,
                                   NativeCpuRuntimeContext,
                                   Scheduler::Context {
    Result define() override {
        const auto taint = skipTestState().taintFor(name());
        if (taint != Module::Taint::CLEAN) {
            JST_CHECK(defineTaint(taint));
        }

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
        if (state.delayFor(name()).count() > 0) {
            std::this_thread::sleep_for(state.delayFor(name()));
        }
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
        const auto taint = skipTestState().taintFor(name());
        if (taint != Module::Taint::CLEAN) {
            JST_CHECK(defineTaint(taint));
        }

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

struct SkipTestFailInitModule : Module::Impl,
                                DynamicConfig<SkipTestFailInitConfig>,
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

    Result computeInitialize() override {
        JST_ERROR("[SKIP_TEST_FAIL_INIT] Forced runtime initialization failure.");
        return Result::ERROR;
    }

    Result computeSubmit() override {
        skipTestState().noteCall(name());
        return Result::SUCCESS;
    }

    Tensor output;
};

struct SkipTestPythonModule : Module::Impl,
                              DynamicConfig<SkipTestPythonConfig>,
                              PythonRuntimeContext,
                              Scheduler::Context {
    Result define() override {
        const auto taint = skipTestState().taintFor(name());
        if (taint != Module::Taint::CLEAN) {
            JST_CHECK(defineTaint(taint));
        }

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

JST_REGISTER_MODULE(SkipTestSourceModule, DeviceType::CPU, RuntimeType::NATIVE, "generic");
JST_REGISTER_MODULE(SkipTestPassthroughModule, DeviceType::CPU, RuntimeType::NATIVE, "generic");
JST_REGISTER_MODULE(SkipTestMergeModule, DeviceType::CPU, RuntimeType::NATIVE, "generic");
JST_REGISTER_MODULE(SkipTestFailInitModule, DeviceType::CPU, RuntimeType::NATIVE, "generic");
JST_REGISTER_MODULE(SkipTestPythonModule, DeviceType::CPU, RuntimeType::PYTHON, "generic");

Tensor makeTensor() {
    Tensor tensor;

    if (tensor.create(DeviceType::CPU, DataType::F32, {1}) != Result::SUCCESS) {
        throw std::runtime_error("failed to create test tensor");
    }

    tensor.at<F32>(0) = 1.0f;
    return tensor;
}

TensorMap makeInputs(std::initializer_list<std::pair<std::string, std::string>> links) {
    TensorMap inputs;
    for (const auto& [slot, producer] : links) {
        inputs[slot].produced(producer, "out", makeTensor());
    }
    return inputs;
}

std::shared_ptr<Module> createModule(const std::string& type,
                                      const std::string& name,
                                      const TensorMap& inputs = {},
                                      RuntimeType runtime = RuntimeType::NATIVE) {
    std::shared_ptr<Module> module;

    if (Registry::BuildModule(type, DeviceType::CPU, runtime, "generic", module) != Result::SUCCESS) {
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

    std::unordered_set<std::string> failedModules;
    REQUIRE(scheduler.compute(failedModules) == Result::SUCCESS);

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

TEST_CASE("Scheduler preserves runtimes when add fails during runtime initialization", "[runtime][scheduler][rollback]") {
    auto& state = skipTestState();
    state.reset();

    Scheduler scheduler(SchedulerType::SYNCHRONOUS);
    REQUIRE(scheduler.create(nullptr) == Result::SUCCESS);

    auto source = createModule("skip_test_source", "rollback_source");
    auto failInit = createModule("skip_test_fail_init", "rollback_fail_init");

    std::unordered_set<std::string> failedModules;
    REQUIRE(scheduler.add(source) == Result::SUCCESS);
    REQUIRE(scheduler.compute(failedModules) == Result::SUCCESS);
    REQUIRE(state.callCount("rollback_source") == 1);

    REQUIRE(scheduler.add(failInit) == Result::ERROR);

    REQUIRE(scheduler.compute(failedModules) == Result::SUCCESS);
    REQUIRE(state.callCount("rollback_source") == 2);
    REQUIRE(state.callCount("rollback_fail_init") == 0);

    REQUIRE(scheduler.remove(source) == Result::SUCCESS);
    REQUIRE(scheduler.destroy() == Result::SUCCESS);
    REQUIRE(failInit->destroy() == Result::SUCCESS);
    REQUIRE(source->destroy() == Result::SUCCESS);
}

TEST_CASE("Runtime cleans up partial initialization failures", "[runtime][rollback]") {
    auto& state = skipTestState();
    state.reset();
    state.setInitializeFailureAt(2);

    auto sourceA = createModule("skip_test_source", "partial_init_source_a");
    auto sourceB = createModule("skip_test_source", "partial_init_source_b");

    Runtime runtime("partial_init", DeviceType::CPU, RuntimeType::NATIVE);
    REQUIRE(runtime.create({
        {"partial_init_source_a", sourceA},
        {"partial_init_source_b", sourceB},
    }) == Result::ERROR);

    REQUIRE(state.initializeCount == 2);
    REQUIRE(state.totalDeinitializeCallCount() == 2);
    REQUIRE(state.deinitializeCallCount("partial_init_source_a") ==
            state.initializeCallCount("partial_init_source_a"));
    REQUIRE(state.deinitializeCallCount("partial_init_source_b") ==
            state.initializeCallCount("partial_init_source_b"));

    REQUIRE(runtime.destroy() == Result::SUCCESS);
    REQUIRE(sourceB->destroy() == Result::SUCCESS);
    REQUIRE(sourceA->destroy() == Result::SUCCESS);
}

TEST_CASE("Runtime reports failed modules", "[runtime][failure]") {
    auto& state = skipTestState();
    state.reset();

    auto source = createModule("skip_test_source", "failed_module_source");

    Runtime runtime("failed_module", DeviceType::CPU, RuntimeType::NATIVE);
    REQUIRE(runtime.create({{"failed_module_source", source}}) == Result::SUCCESS);

    state.setResult("failed_module_source", Result::ERROR);

    std::unordered_set<std::string> skippedModules;
    std::unordered_set<std::string> failedModules;
    REQUIRE(runtime.compute({}, skippedModules, failedModules) == Result::ERROR);
    REQUIRE(failedModules.contains("failed_module_source"));

    REQUIRE(runtime.destroy() == Result::SUCCESS);
    REQUIRE(source->destroy() == Result::SUCCESS);
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
    std::unordered_set<std::string> failedModules;
    REQUIRE(upstream.compute({"barrier_source"}, skippedModules, failedModules) == Result::SUCCESS);
    REQUIRE(downstream.compute({"barrier_sink"}, skippedModules, failedModules) == Result::SUCCESS);

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
    std::unordered_set<std::string> failedModules;
    REQUIRE(runtime.compute({
        "fanin_source_skip",
        "fanin_source_run",
        "fanin_merge",
    }, skippedModules, failedModules) == Result::SUCCESS);

    REQUIRE(state.callCount("fanin_source_skip") == 1);
    REQUIRE(state.callCount("fanin_source_run") == 1);
    REQUIRE(state.callCount("fanin_merge") == 0);

    REQUIRE(runtime.destroy() == Result::SUCCESS);
    REQUIRE(merge->destroy() == Result::SUCCESS);
    REQUIRE(sourceRun->destroy() == Result::SUCCESS);
    REQUIRE(sourceSkip->destroy() == Result::SUCCESS);
}

TEST_CASE("Module composes static scheduler taints", "[runtime][scheduler][static][taint]") {
    auto& state = skipTestState();
    state.reset();
    state.setTaint("static_taint_source",
                   Module::Taint::STATIC_OUTPUT |
                       Module::Taint::STATELESS |
                       Module::Taint::DISCONTIGUOUS);

    auto source = createModule("skip_test_source", "static_taint_source");
    REQUIRE((source->taint() & Module::Taint::STATIC_OUTPUT) ==
            Module::Taint::STATIC_OUTPUT);
    REQUIRE((source->taint() & Module::Taint::STATELESS) ==
            Module::Taint::STATELESS);
    REQUIRE((source->taint() & Module::Taint::DISCONTIGUOUS) ==
            Module::Taint::DISCONTIGUOUS);
    REQUIRE(source->destroy() == Result::SUCCESS);
}

TEST_CASE("Scheduler settles fully static branches without stopping mixed inputs",
          "[runtime][scheduler][static]") {
    auto& state = skipTestState();
    state.reset();
    state.setTaint("static_left", Module::Taint::STATIC_OUTPUT);
    state.setTaint("static_transform", Module::Taint::STATELESS);
    state.setTaint("static_right", Module::Taint::STATIC_OUTPUT);
    state.setTaint("static_merge", Module::Taint::STATELESS);
    state.setTaint("mixed_merge", Module::Taint::STATELESS);
    state.setTaint("mixed_sink", Module::Taint::STATELESS);

    auto staticLeft = createModule("skip_test_source", "static_left");
    auto staticTransform = createModule(
        "skip_test_passthrough",
        "static_transform",
        makeInputs({{"in", "static_left"}}));
    auto staticRight = createModule("skip_test_source", "static_right");
    auto staticMerge = createModule(
        "skip_test_merge",
        "static_merge",
        makeInputs({{"left", "static_transform"}, {"right", "static_right"}}));
    auto liveSource = createModule("skip_test_source", "live_source");
    auto mixedMerge = createModule(
        "skip_test_merge",
        "mixed_merge",
        makeInputs({{"left", "static_merge"}, {"right", "live_source"}}));
    auto mixedSink = createModule(
        "skip_test_passthrough",
        "mixed_sink",
        makeInputs({{"in", "mixed_merge"}}));

    std::vector<std::shared_ptr<Module>> modules = {
        staticLeft,
        staticTransform,
        staticRight,
        staticMerge,
        liveSource,
        mixedMerge,
        mixedSink,
    };
    Scheduler scheduler(SchedulerType::SYNCHRONOUS);
    REQUIRE(scheduler.create(nullptr) == Result::SUCCESS);
    for (const auto& module : modules) {
        REQUIRE(scheduler.add(module) == Result::SUCCESS);
    }

    std::unordered_set<std::string> failedModules;
    for (U64 cycle = 0; cycle < 3; ++cycle) {
        REQUIRE(scheduler.compute(failedModules) == Result::SUCCESS);
        REQUIRE(failedModules.empty());
    }

    REQUIRE(state.callCount("static_left") == 1);
    REQUIRE(state.callCount("static_transform") == 1);
    REQUIRE(state.callCount("static_right") == 1);
    REQUIRE(state.callCount("static_merge") == 1);
    REQUIRE(state.callCount("live_source") == 3);
    REQUIRE(state.callCount("mixed_merge") == 3);
    REQUIRE(state.callCount("mixed_sink") == 3);
    REQUIRE(staticLeft->timing().computeTime == 0.0f);
    REQUIRE(staticTransform->timing().computeTime == 0.0f);
    REQUIRE(staticMerge->timing().computeTime == 0.0f);

    REQUIRE(scheduler.destroy() == Result::SUCCESS);
    destroyModules(modules);
}

TEST_CASE("Scheduler retries static modules until output is usable",
          "[runtime][scheduler][static]") {
    for (const auto initialResult : {Result::SKIP, Result::YIELD, Result::TIMEOUT}) {
        DYNAMIC_SECTION("Initial result: " << initialResult) {
            auto& state = skipTestState();
            state.reset();
            state.setTaint("retry_static_source", Module::Taint::STATIC_OUTPUT);
            state.setTaint("retry_static_sink", Module::Taint::STATELESS);
            state.setResults("retry_static_source", {initialResult, Result::SUCCESS});

            auto source = createModule("skip_test_source", "retry_static_source");
            auto sink = createModule(
                "skip_test_passthrough",
                "retry_static_sink",
                makeInputs({{"in", "retry_static_source"}}));
            std::vector<std::shared_ptr<Module>> modules = {source, sink};

            Scheduler scheduler(SchedulerType::SYNCHRONOUS);
            REQUIRE(scheduler.create(nullptr) == Result::SUCCESS);
            REQUIRE(scheduler.add(source) == Result::SUCCESS);
            REQUIRE(scheduler.add(sink) == Result::SUCCESS);

            std::unordered_set<std::string> failedModules;
            REQUIRE(scheduler.compute(failedModules) == Result::SUCCESS);
            REQUIRE(scheduler.compute(failedModules) == Result::SUCCESS);
            REQUIRE(scheduler.compute(failedModules) == Result::SUCCESS);
            REQUIRE(state.callCount("retry_static_source") == 2);
            REQUIRE(state.callCount("retry_static_sink") == 1);

            REQUIRE(scheduler.destroy() == Result::SUCCESS);
            destroyModules(modules);
        }
    }
}

TEST_CASE("Scheduler invalidates settlement only for state mutations",
          "[runtime][scheduler][static]") {
    auto& state = skipTestState();

    SECTION("stop and start preserve settlement while synchronize clears it") {
        state.reset();
        state.setTaint("lifecycle_static", Module::Taint::STATIC_OUTPUT);
        state.setDelay("lifecycle_static", std::chrono::milliseconds(1));

        auto source = createModule("skip_test_source", "lifecycle_static");
        std::vector<std::shared_ptr<Module>> modules = {source};
        Scheduler scheduler(SchedulerType::SYNCHRONOUS);
        REQUIRE(scheduler.create(nullptr) == Result::SUCCESS);
        REQUIRE(scheduler.add(source) == Result::SUCCESS);

        std::unordered_set<std::string> failedModules;
        REQUIRE(scheduler.compute(failedModules) == Result::SUCCESS);
        REQUIRE(scheduler.compute(failedModules) == Result::SUCCESS);
        REQUIRE(state.callCount("lifecycle_static") == 1);

        REQUIRE(scheduler.stop() == Result::SUCCESS);
        REQUIRE(scheduler.start() == Result::SUCCESS);
        REQUIRE(scheduler.compute(failedModules) == Result::SUCCESS);
        REQUIRE(state.callCount("lifecycle_static") == 1);

        REQUIRE(scheduler.synchronize([] { return Result::SUCCESS; }) == Result::SUCCESS);
        REQUIRE(scheduler.compute(failedModules) == Result::SUCCESS);
        REQUIRE(state.callCount("lifecycle_static") == 2);

        REQUIRE(scheduler.destroy() == Result::SUCCESS);
        REQUIRE(source->timing().computeTime > 0.0f);
        destroyModules(modules);
    }

    SECTION("input mutation disables settlement for the whole graph") {
        state.reset();
        state.setTaint("in_place_static", Module::Taint::STATIC_OUTPUT);
        state.setTaint("in_place_consumer", Module::Taint::IN_PLACE);

        auto source = createModule("skip_test_source", "in_place_static");
        auto consumer = createModule(
            "skip_test_passthrough",
            "in_place_consumer",
            makeInputs({{"in", "in_place_static"}}));
        std::vector<std::shared_ptr<Module>> modules = {source, consumer};
        Scheduler scheduler(SchedulerType::SYNCHRONOUS);
        REQUIRE(scheduler.create(nullptr) == Result::SUCCESS);
        REQUIRE(scheduler.add(source) == Result::SUCCESS);
        REQUIRE(scheduler.add(consumer) == Result::SUCCESS);

        std::unordered_set<std::string> failedModules;
        REQUIRE(scheduler.compute(failedModules) == Result::SUCCESS);
        REQUIRE(scheduler.compute(failedModules) == Result::SUCCESS);
        REQUIRE(state.callCount("in_place_static") == 2);
        REQUIRE(state.callCount("in_place_consumer") == 2);

        REQUIRE(scheduler.destroy() == Result::SUCCESS);
        destroyModules(modules);
    }
}

TEST_CASE("Scheduler keeps settled surfaces presentable across compute cycles",
          "[runtime][scheduler][static]") {
    auto& state = skipTestState();
    state.reset();
    state.setSurface("static_surface");
    state.setTaint("static_surface",
                   Module::Taint::STATIC_OUTPUT | Module::Taint::THROTTLED);

    auto surface = createModule("skip_test_source", "static_surface");
    auto sink = createModule(
        "skip_test_passthrough",
        "static_surface_sink",
        makeInputs({{"in", "static_surface"}}));
    std::vector<std::shared_ptr<Module>> modules = {surface, sink};
    Scheduler scheduler(SchedulerType::SYNCHRONOUS);
    REQUIRE(scheduler.create(nullptr) == Result::SUCCESS);
    REQUIRE(scheduler.add(surface) == Result::SUCCESS);
    REQUIRE(scheduler.add(sink) == Result::SUCCESS);

    std::unordered_set<std::string> failedModules;
    REQUIRE(scheduler.compute(failedModules) == Result::SUCCESS);
    REQUIRE(scheduler.compute(failedModules) == Result::SUCCESS);
    REQUIRE(scheduler.present(failedModules) == Result::SUCCESS);
    REQUIRE(scheduler.present(failedModules) == Result::SUCCESS);
    REQUIRE(state.pendingCallCount("static_surface") == 1);
    REQUIRE(state.callCount("static_surface") == 1);
    REQUIRE(state.callCount("static_surface_sink") == 2);
    REQUIRE(state.presentCallCount("static_surface") == 2);

    REQUIRE(scheduler.destroy() == Result::SUCCESS);
    destroyModules(modules);
}

TEST_CASE("Scheduler preserves settlement across runtime segments",
          "[runtime][scheduler][static]") {
    auto& state = skipTestState();
    state.reset();
    state.setTaint("boundary_native_source", Module::Taint::STATIC_OUTPUT);
    state.setTaint("boundary_python", Module::Taint::STATELESS);
    state.setResults("boundary_native_source", {Result::YIELD, Result::SUCCESS});

    auto source = createModule("skip_test_source", "boundary_native_source");
    auto python = createModule(
        "skip_test_python",
        "boundary_python",
        makeInputs({{"in", "boundary_native_source"}}),
        RuntimeType::PYTHON);
    auto sink = createModule(
        "skip_test_passthrough",
        "boundary_native_sink",
        makeInputs({{"in", "boundary_python"}}));
    std::vector<std::shared_ptr<Module>> modules = {source, python, sink};
    Scheduler scheduler(SchedulerType::SYNCHRONOUS);
    REQUIRE(scheduler.create(nullptr) == Result::SUCCESS);
    REQUIRE(scheduler.add(source) == Result::SUCCESS);
    REQUIRE(scheduler.add(python) == Result::SUCCESS);
    REQUIRE(scheduler.add(sink) == Result::SUCCESS);

    std::unordered_set<std::string> failedModules;
    REQUIRE(scheduler.compute(failedModules) == Result::SUCCESS);
    REQUIRE(state.callCount("boundary_native_source") == 1);
    REQUIRE(state.callCount("boundary_python") == 0);
    REQUIRE(state.callCount("boundary_native_sink") == 0);
    REQUIRE(scheduler.compute(failedModules) == Result::SUCCESS);
    REQUIRE(scheduler.compute(failedModules) == Result::SUCCESS);
    REQUIRE(state.callCount("boundary_native_source") == 2);
    REQUIRE(state.callCount("boundary_python") == 1);
    REQUIRE(state.callCount("boundary_native_sink") == 2);

    REQUIRE(scheduler.destroy() == Result::SUCCESS);
    destroyModules(modules);
}
