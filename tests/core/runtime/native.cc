#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "jetstream/detail/module_impl.hh"
#include "jetstream/module_context.hh"
#include "jetstream/registry.hh"
#include "jetstream/runtime_context_native_cpu.hh"
#include "jetstream/scheduler.hh"
#include "jetstream/scheduler_context.hh"

namespace {

using namespace Jetstream;

struct SkipTestState {
    std::unordered_map<std::string, std::vector<Result>> results;
    std::unordered_map<std::string, U64> resultCursors;
    std::unordered_map<std::string, U64> calls;
    std::unordered_map<std::string, U64> initializes;
    std::unordered_map<std::string, U64> deinitializes;
    std::unordered_map<std::string, std::vector<Result>> deinitializeResults;
    std::unordered_map<std::string, U64> deinitializeResultCursors;
    std::vector<std::string> submissions;
    std::vector<std::string> lifecycleEvents;
    U64 initializeCount = 0;
    U64 failInitializeAt = 0;

    void reset() {
        results.clear();
        resultCursors.clear();
        calls.clear();
        initializes.clear();
        deinitializes.clear();
        deinitializeResults.clear();
        deinitializeResultCursors.clear();
        submissions.clear();
        lifecycleEvents.clear();
        initializeCount = 0;
        failInitializeAt = 0;
    }

    void setResult(const std::string& module, const Result result) {
        results[module] = {result};
    }

    void setResults(const std::string& module, std::vector<Result> moduleResults) {
        results[module] = std::move(moduleResults);
    }

    Result resultFor(const std::string& module) {
        if (!results.contains(module)) {
            return Result::SUCCESS;
        }

        auto& cursor = resultCursors[module];
        const auto& moduleResults = results.at(module);
        if (cursor == moduleResults.size()) {
            return Result::SUCCESS;
        }

        return moduleResults[cursor++];
    }

    void noteCall(const std::string& module) {
        calls[module] += 1;
        submissions.push_back(module);
    }

    U64 callCount(const std::string& module) const {
        if (!calls.contains(module)) {
            return 0;
        }

        return calls.at(module);
    }

    void setInitializeFailureAt(const U64 count) {
        failInitializeAt = count;
    }

    Result noteInitialize(const std::string& module) {
        initializes[module] += 1;
        initializeCount += 1;
        lifecycleEvents.push_back("initialize:" + module);

        if (failInitializeAt != 0 && initializeCount >= failInitializeAt) {
            return Result::ERROR;
        }

        return Result::SUCCESS;
    }

    Result noteDeinitialize(const std::string& module) {
        deinitializes[module] += 1;
        lifecycleEvents.push_back("deinitialize:" + module);

        if (!deinitializeResults.contains(module)) {
            return Result::SUCCESS;
        }

        auto& cursor = deinitializeResultCursors[module];
        const auto& moduleResults = deinitializeResults.at(module);
        if (cursor == moduleResults.size()) {
            return Result::SUCCESS;
        }

        return moduleResults[cursor++];
    }

    void setDeinitializeResult(const std::string& module, const Result result) {
        deinitializeResults[module] = {result};
    }

    void clearLifecycleEvents() {
        lifecycleEvents.clear();
    }

    std::vector<std::string> lifecycleCalls(const std::string& operation) const {
        std::vector<std::string> modules;
        const std::string prefix = operation + ":";

        for (const auto& event : lifecycleEvents) {
            if (event.starts_with(prefix)) {
                modules.push_back(event.substr(prefix.size()));
            }
        }

        return modules;
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

struct NativeContextDefaultConfig : Module::Config {
    JST_MODULE_TYPE(native_context_default)

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

struct IncompatibleContextConfig : Module::Config {
    JST_MODULE_TYPE(incompatible_native_context)

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

    Result computeInitialize() override {
        const auto result = skipTestState().noteInitialize(name());
        if (result != Result::SUCCESS) {
            JST_ERROR("[SKIP_TEST_SOURCE] Forced runtime initialization failure.");
        }

        return result;
    }

    Result computeDeinitialize() override {
        return skipTestState().noteDeinitialize(name());
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

struct NativeContextDefaultModule : Module::Impl,
                                    DynamicConfig<NativeContextDefaultConfig>,
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

    Tensor output;
};

struct IncompatibleContextModule : Module::Impl,
                                   DynamicConfig<IncompatibleContextConfig>,
                                   Scheduler::Context {
    Result define() override {
        JST_CHECK(defineInterfaceOutput("out"));
        return Result::SUCCESS;
    }

    Result create() override {
        JST_CHECK(output.create(DeviceType::CPU, DataType::F32, {1}));
        outputs()["out"].produced(name(), "out", output);
        return Result::SUCCESS;
    }

    Tensor output;
};

JST_REGISTER_MODULE(SkipTestSourceModule, DeviceType::CPU, RuntimeType::NATIVE, "generic");
JST_REGISTER_MODULE(SkipTestPassthroughModule, DeviceType::CPU, RuntimeType::NATIVE, "generic");
JST_REGISTER_MODULE(SkipTestMergeModule, DeviceType::CPU, RuntimeType::NATIVE, "generic");
JST_REGISTER_MODULE(NativeContextDefaultModule, DeviceType::CPU, RuntimeType::NATIVE, "generic");

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

std::shared_ptr<Module> createIncompatibleContextModule(const std::string& name,
                                                        const bool nullContext) {
    auto impl = std::make_shared<IncompatibleContextModule>();
    std::shared_ptr<Module::Context> context;

    if (!nullContext) {
        auto runtimeContext = std::make_shared<Runtime::Context>();
        auto schedulerContext = std::static_pointer_cast<Scheduler::Context>(impl);
        context = std::make_shared<Module::Context>(runtimeContext,
                                                    schedulerContext,
                                                    nullptr,
                                                    nullptr);
    }

    auto stagedConfig = std::static_pointer_cast<Module::Config>(impl);
    auto candidateConfig = std::static_pointer_cast<Module::Config>(impl->candidate());
    auto module = std::make_shared<Module>(DeviceType::CPU,
                                           RuntimeType::MLIR,
                                           "generic",
                                           impl,
                                           context,
                                           stagedConfig,
                                           candidateConfig);
    Parser::Map config;
    if (module->create(name, config, {}) != Result::SUCCESS) {
        throw std::runtime_error("failed to create incompatible context module: " + name);
    }

    return module;
}

void destroyModules(std::vector<std::shared_ptr<Module>>& modules) {
    for (auto it = modules.rbegin(); it != modules.rend(); ++it) {
        REQUIRE((*it)->destroy() == Result::SUCCESS);
    }
}

}  // namespace

TEST_CASE("Native context defaults provide no-op runtime lifecycle", "[core][runtime][native]") {
    skipTestState().reset();
    auto module = createModule("native_context_default", "native_context_default");

    Runtime runtime("native_context_default", DeviceType::CPU, RuntimeType::NATIVE);
    REQUIRE(runtime.create({{"native_context_default", module}}) == Result::SUCCESS);

    std::unordered_set<std::string> skippedModules;
    std::unordered_set<std::string> failedModules = {"stale"};
    REQUIRE(runtime.compute({"native_context_default"}, skippedModules, failedModules) == Result::SUCCESS);
    REQUIRE(skippedModules.empty());
    REQUIRE(failedModules.empty());
    REQUIRE(module->timing().cycles == 1);

    REQUIRE(runtime.destroy() == Result::SUCCESS);
    REQUIRE(module->destroy() == Result::SUCCESS);
}

TEST_CASE("Native runtime follows the requested submission order", "[core][runtime][native]") {
    auto& state = skipTestState();
    state.reset();

    auto first = createModule("skip_test_source", "order_first");
    auto second = createModule("skip_test_source", "order_second");
    auto third = createModule("skip_test_source", "order_third");
    std::vector<std::shared_ptr<Module>> modules = {first, second, third};

    Runtime runtime("order", DeviceType::CPU, RuntimeType::NATIVE);
    REQUIRE(runtime.create({
        {"order_first", first},
        {"order_second", second},
        {"order_third", third},
    }) == Result::SUCCESS);

    std::unordered_set<std::string> skippedModules;
    std::unordered_set<std::string> failedModules = {"stale"};
    REQUIRE(runtime.compute({"order_third", "order_first", "order_second"},
                            skippedModules,
                            failedModules) == Result::SUCCESS);
    REQUIRE(skippedModules.empty());
    REQUIRE(failedModules.empty());
    REQUIRE(state.submissions == std::vector<std::string>{
                                     "order_third", "order_first", "order_second"});

    REQUIRE(runtime.destroy() == Result::SUCCESS);
    destroyModules(modules);
}

TEST_CASE("Native runtime resumes YIELD and TIMEOUT submissions", "[core][runtime][native]") {
    for (const auto status : {Result::YIELD, Result::TIMEOUT}) {
        CAPTURE(status);
        auto& state = skipTestState();
        state.reset();
        state.setResults("poll_source", {status, Result::SUCCESS});

        auto source = createModule("skip_test_source", "poll_source");
        TensorMap sinkInputs;
        sinkInputs["in"].produced("poll_source", "out", makeTensor());
        auto sink = createModule("skip_test_passthrough", "poll_sink", sinkInputs);
        std::vector<std::shared_ptr<Module>> modules = {source, sink};

        Runtime runtime("poll", DeviceType::CPU, RuntimeType::NATIVE);
        REQUIRE(runtime.create({{"poll_source", source}, {"poll_sink", sink}}) == Result::SUCCESS);

        std::unordered_set<std::string> skippedModules;
        std::unordered_set<std::string> failedModules;
        REQUIRE(runtime.compute({"poll_source", "poll_sink"}, skippedModules, failedModules) == status);
        REQUIRE(failedModules.empty());
        REQUIRE(state.submissions == std::vector<std::string>{"poll_source"});

        REQUIRE(runtime.compute({"poll_source", "poll_sink"}, skippedModules, failedModules) == Result::SUCCESS);
        REQUIRE(skippedModules.empty());
        REQUIRE(failedModules.empty());
        REQUIRE(state.submissions == std::vector<std::string>{
                                         "poll_source", "poll_source", "poll_sink"});

        REQUIRE(runtime.destroy() == Result::SUCCESS);
        destroyModules(modules);
    }
}

TEST_CASE("Native runtime cleans up partial initialization failures", "[core][runtime][native]") {
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

TEST_CASE("Native runtime stops and reports the failed module", "[core][runtime][native]") {
    auto& state = skipTestState();
    state.reset();

    auto source = createModule("skip_test_source", "failed_module_source");
    auto later = createModule("skip_test_source", "failed_module_later");

    Runtime runtime("failed_module", DeviceType::CPU, RuntimeType::NATIVE);
    REQUIRE(runtime.create({
        {"failed_module_source", source},
        {"failed_module_later", later},
    }) == Result::SUCCESS);

    state.setResult("failed_module_source", Result::ERROR);

    std::unordered_set<std::string> skippedModules;
    std::unordered_set<std::string> failedModules = {"stale"};
    REQUIRE(runtime.compute({"failed_module_source", "failed_module_later"},
                            skippedModules,
                            failedModules) == Result::ERROR);
    REQUIRE(skippedModules.empty());
    REQUIRE(failedModules == std::unordered_set<std::string>{"failed_module_source"});
    REQUIRE(state.submissions == std::vector<std::string>{"failed_module_source"});

    REQUIRE(runtime.destroy() == Result::SUCCESS);
    REQUIRE(later->destroy() == Result::SUCCESS);
    REQUIRE(source->destroy() == Result::SUCCESS);
}

TEST_CASE("Native runtime propagates SKIP across compute barriers", "[core][runtime][native]") {
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
    REQUIRE(skippedModules == std::unordered_set<std::string>{"barrier_source", "barrier_sink"});
    REQUIRE(failedModules.empty());

    REQUIRE(downstream.destroy() == Result::SUCCESS);
    REQUIRE(upstream.destroy() == Result::SUCCESS);
    REQUIRE(sink->destroy() == Result::SUCCESS);
    REQUIRE(source->destroy() == Result::SUCCESS);
}

TEST_CASE("Native runtime skips fan-in consumers when any input skips", "[core][runtime][native]") {
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
    REQUIRE(skippedModules == std::unordered_set<std::string>{"fanin_source_skip", "fanin_merge"});
    REQUIRE(failedModules.empty());

    REQUIRE(runtime.destroy() == Result::SUCCESS);
    REQUIRE(merge->destroy() == Result::SUCCESS);
    REQUIRE(sourceRun->destroy() == Result::SUCCESS);
    REQUIRE(sourceSkip->destroy() == Result::SUCCESS);
}

TEST_CASE("Native runtime cleans failed initialization before module destruction",
          "[core][runtime][native]") {
    auto& state = skipTestState();
    state.reset();
    state.setInitializeFailureAt(1);

    auto module = createModule("skip_test_source", "cleanup_failure");
    Runtime runtime("cleanup_failure", DeviceType::CPU, RuntimeType::NATIVE);
    REQUIRE(runtime.create({{"cleanup_failure", module}}) == Result::ERROR);
    REQUIRE(state.initializeCallCount("cleanup_failure") == 1);
    REQUIRE(state.deinitializeCallCount("cleanup_failure") == 1);
    REQUIRE(runtime.destroy() == Result::SUCCESS);
    REQUIRE(module->destroy() == Result::SUCCESS);
}

TEST_CASE("Native runtime tears modules down in reverse initialization order",
          "[core][runtime][native]") {
    auto& state = skipTestState();
    state.reset();

    auto first = createModule("skip_test_source", "teardown_first");
    auto second = createModule("skip_test_source", "teardown_second");
    auto third = createModule("skip_test_source", "teardown_third");
    std::vector<std::shared_ptr<Module>> modules = {first, second, third};

    Runtime runtime("teardown_order", DeviceType::CPU, RuntimeType::NATIVE);
    REQUIRE(runtime.create({
        {"teardown_first", first},
        {"teardown_second", second},
        {"teardown_third", third},
    }) == Result::SUCCESS);

    auto expected = state.lifecycleCalls("initialize");
    std::ranges::reverse(expected);
    state.clearLifecycleEvents();

    REQUIRE(runtime.destroy() == Result::SUCCESS);
    // Current defect: Native CPU teardown follows unordered storage instead of reverse initialization order.
    CHECK(state.lifecycleCalls("deinitialize") == expected);
    destroyModules(modules);
}

TEST_CASE("Native runtime continues teardown after deinitialization failures",
          "[core][runtime][native]") {
    auto& state = skipTestState();
    state.reset();

    auto first = createModule("skip_test_source", "teardown_error_first");
    auto second = createModule("skip_test_source", "teardown_error_second");
    auto third = createModule("skip_test_source", "teardown_error_third");
    std::vector<std::shared_ptr<Module>> modules = {first, second, third};

    Runtime runtime("teardown_error", DeviceType::CPU, RuntimeType::NATIVE);
    REQUIRE(runtime.create({
        {"teardown_error_first", first},
        {"teardown_error_second", second},
        {"teardown_error_third", third},
    }) == Result::SUCCESS);

    state.setDeinitializeResult("teardown_error_first", Result::ERROR);
    state.setDeinitializeResult("teardown_error_second", Result::ERROR);
    state.setDeinitializeResult("teardown_error_third", Result::ERROR);
    state.clearLifecycleEvents();

    REQUIRE(runtime.destroy() == Result::ERROR);
    // Current defect: Native CPU teardown returns after the first failure and leaves later contexts initialized.
    CHECK(state.totalDeinitializeCallCount() == 3);
    destroyModules(modules);
}

TEST_CASE("Native runtime supports repeated create and destroy cycles",
          "[core][runtime][native]") {
    auto& state = skipTestState();
    state.reset();

    auto module = createModule("skip_test_source", "repeated_lifecycle");
    Runtime runtime("repeated_lifecycle", DeviceType::CPU, RuntimeType::NATIVE);

    for (U64 cycle = 1; cycle <= 2; ++cycle) {
        CAPTURE(cycle);
        REQUIRE(runtime.create({{"repeated_lifecycle", module}}) == Result::SUCCESS);
        REQUIRE(state.initializeCallCount("repeated_lifecycle") == cycle);
        REQUIRE(runtime.destroy() == Result::SUCCESS);
        REQUIRE(state.deinitializeCallCount("repeated_lifecycle") == cycle);
        REQUIRE(runtime.destroy() == Result::SUCCESS);
        REQUIRE(state.deinitializeCallCount("repeated_lifecycle") == cycle);
    }

    REQUIRE(module->destroy() == Result::SUCCESS);
}

TEST_CASE("Native runtime rejects incompatible modules before accessing invalid contexts",
          "[core][runtime][native]") {
    for (const bool nullContext : {false, true}) {
        CAPTURE(nullContext);
        const std::string name = nullContext ? "null_context" : "wrong_context";
        auto module = createIncompatibleContextModule(name, nullContext);
        Runtime runtime(name, DeviceType::CPU, RuntimeType::NATIVE);

        REQUIRE(runtime.create({{name, module}}) == Result::ERROR);
        REQUIRE(runtime.destroy() == Result::SUCCESS);
        REQUIRE(module->destroy() == Result::SUCCESS);
    }
}

TEST_CASE("Native runtime reports unknown requested module contexts",
          "[core][runtime][native]") {
    auto& state = skipTestState();
    state.reset();

    auto module = createModule("skip_test_source", "known_context");
    Runtime runtime("unknown_context", DeviceType::CPU, RuntimeType::NATIVE);
    REQUIRE(runtime.create({{"known_context", module}}) == Result::SUCCESS);

    std::unordered_set<std::string> skippedModules;
    std::unordered_set<std::string> failedModules = {"stale"};
    REQUIRE(runtime.compute({"missing_context"}, skippedModules, failedModules) == Result::ERROR);
    REQUIRE(failedModules == std::unordered_set<std::string>{"missing_context"});
    REQUIRE(state.submissions.empty());

    REQUIRE(runtime.destroy() == Result::SUCCESS);
    REQUIRE(module->destroy() == Result::SUCCESS);
}

TEST_CASE("Native runtime treats RELOAD as successful progress",
          "[core][runtime][native]") {
    auto& state = skipTestState();
    state.reset();
    state.setResult("reload_source", Result::RELOAD);

    auto source = createModule("skip_test_source", "reload_source");
    auto later = createModule("skip_test_source", "reload_later");
    std::vector<std::shared_ptr<Module>> modules = {source, later};

    Runtime runtime("reload", DeviceType::CPU, RuntimeType::NATIVE);
    REQUIRE(runtime.create({{"reload_source", source}, {"reload_later", later}}) == Result::SUCCESS);

    std::unordered_set<std::string> skippedModules;
    std::unordered_set<std::string> failedModules = {"stale"};
    REQUIRE(runtime.compute({"reload_source", "reload_later"},
                            skippedModules,
                            failedModules) == Result::SUCCESS);
    REQUIRE(skippedModules.empty());
    REQUIRE(failedModules.empty());
    REQUIRE(state.submissions == std::vector<std::string>{"reload_source", "reload_later"});

    REQUIRE(runtime.destroy() == Result::SUCCESS);
    destroyModules(modules);
}
