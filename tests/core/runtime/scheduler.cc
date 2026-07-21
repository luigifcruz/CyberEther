#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <functional>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "jetstream/detail/module_impl.hh"
#include "jetstream/module_context.hh"
#include "jetstream/runtime.hh"
#include "jetstream/runtime_context_native_cpu.hh"
#include "jetstream/runtime_context_python.hh"
#include "jetstream/scheduler.hh"
#include "jetstream/scheduler_context.hh"

namespace {

using namespace Jetstream;

struct SyntheticSpec {
    std::vector<std::string> inputs;
    bool surface = false;
    Module::Taint taint = Module::Taint::CLEAN;
    std::chrono::milliseconds computeDelay{0};
    std::vector<Result> initializeResults;
    std::vector<Result> deinitializeResults;
    std::vector<Result> pendingComputeResults;
    std::vector<Result> computeResults;
    std::vector<Result> presentInitializeResults;
    std::vector<Result> presentResults;
    std::function<void()> pendingAction;
};

struct SyntheticState {
    std::unordered_map<std::string, SyntheticSpec> specs;
    std::unordered_map<std::string, std::size_t> cursors;
    std::unordered_map<std::string, U64> counts;
    std::vector<std::string> events;

    void reset() {
        specs.clear();
        cursors.clear();
        counts.clear();
        events.clear();
    }

    void clearEvents() {
        counts.clear();
        events.clear();
    }

    SyntheticSpec& configure(const std::string& name) {
        return specs[name];
    }

    const SyntheticSpec& spec(const std::string& name) const {
        return specs.at(name);
    }

    U64 count(const std::string& operation, const std::string& name) const {
        const auto it = counts.find(operation + ":" + name);
        return it == counts.end() ? 0 : it->second;
    }

    std::vector<std::string> calls(const std::string& operation) const {
        std::vector<std::string> names;
        const std::string prefix = operation + ":";

        for (const auto& event : events) {
            if (event.starts_with(prefix)) {
                names.push_back(event.substr(prefix.size()));
            }
        }

        return names;
    }

    Result invoke(const std::string& operation,
                  const std::string& name,
                  const std::vector<Result>& results) {
        const std::string key = operation + ":" + name;
        events.push_back(key);
        counts[key] += 1;

        auto& cursor = cursors[key];
        if (cursor == results.size()) {
            return Result::SUCCESS;
        }

        return results[cursor++];
    }
};

SyntheticState& syntheticState() {
    static SyntheticState state;
    return state;
}

struct SyntheticConfig : Module::Config {
    JST_MODULE_TYPE(scheduler_runtime_synthetic)

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

template<typename RuntimeContext>
struct SyntheticModuleBase : Module::Impl,
                             DynamicConfig<SyntheticConfig>,
                             RuntimeContext,
                             Scheduler::Context {
    Result define() override {
        const auto& spec = syntheticState().spec(name());

        if (spec.surface) {
            JST_CHECK(defineTaint(Module::Taint::SURFACE));
        }
        if (spec.taint != Module::Taint::CLEAN) {
            JST_CHECK(defineTaint(spec.taint));
        }

        for (const auto& input : spec.inputs) {
            JST_CHECK(defineInterfaceInput(input));
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

    Result destroy() override {
        return syntheticState().invoke("destroy", name(), {});
    }

    Result computeInitialize() override {
        const auto& spec = syntheticState().spec(name());
        return syntheticState().invoke("initialize", name(), spec.initializeResults);
    }

    Result computeDeinitialize() override {
        const auto& spec = syntheticState().spec(name());
        return syntheticState().invoke("deinitialize", name(), spec.deinitializeResults);
    }

    Result computeSubmit() override {
        const auto& spec = syntheticState().spec(name());
        if (spec.computeDelay.count() > 0) {
            std::this_thread::sleep_for(spec.computeDelay);
        }
        return syntheticState().invoke("compute", name(), spec.computeResults);
    }

    Result presentInitialize() override {
        const auto& spec = syntheticState().spec(name());
        return syntheticState().invoke("present_initialize", name(), spec.presentInitializeResults);
    }

    Result presentSubmit() override {
        const auto& spec = syntheticState().spec(name());
        return syntheticState().invoke("present", name(), spec.presentResults);
    }

    Result hasPendingCompute() override {
        auto& spec = syntheticState().configure(name());
        if (spec.pendingAction) {
            auto action = std::move(spec.pendingAction);
            spec.pendingAction = {};
            syntheticState().invoke("pending_compute", name(), {});
            action();
            return Result::YIELD;
        }

        return syntheticState().invoke("pending_compute", name(), spec.pendingComputeResults);
    }

    Tensor output;
};

using SyntheticModule = SyntheticModuleBase<NativeCpuRuntimeContext>;
using SyntheticPythonModule = SyntheticModuleBase<PythonRuntimeContext>;

Tensor makeTensor() {
    Tensor tensor;
    if (tensor.create(DeviceType::CPU, DataType::F32, {1}) != Result::SUCCESS) {
        throw std::runtime_error("failed to create scheduler/runtime test tensor");
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

template<typename SyntheticModuleType>
std::shared_ptr<Module> makeModuleForRuntime(const std::string& name,
                                             const TensorMap& inputs,
                                             RuntimeType runtime) {
    auto impl = std::make_shared<SyntheticModuleType>();
    auto runtimeContext = std::static_pointer_cast<Runtime::Context>(impl);
    auto schedulerContext = std::static_pointer_cast<Scheduler::Context>(impl);
    auto context = std::make_shared<Module::Context>(runtimeContext,
                                                     schedulerContext,
                                                     nullptr,
                                                     nullptr);
    auto stagedConfig = std::static_pointer_cast<Module::Config>(impl);
    auto candidateConfig = std::static_pointer_cast<Module::Config>(impl->candidate());
    auto module = std::make_shared<Module>(DeviceType::CPU,
                                           runtime,
                                           "generic",
                                           impl,
                                           context,
                                           stagedConfig,
                                           candidateConfig);
    Parser::Map config;

    if (module->create(name, config, inputs) != Result::SUCCESS) {
        throw std::runtime_error("failed to create synthetic module: " + name);
    }

    return module;
}

std::shared_ptr<Module> makeModule(const std::string& name,
                                   const TensorMap& inputs = {},
                                   RuntimeType runtime = RuntimeType::NATIVE) {
    if (runtime == RuntimeType::NATIVE) {
        return makeModuleForRuntime<SyntheticModule>(name, inputs, runtime);
    }
    if (runtime == RuntimeType::PYTHON) {
        return makeModuleForRuntime<SyntheticPythonModule>(name, inputs, runtime);
    }

    throw std::runtime_error("unsupported synthetic module runtime");
}

struct SchedulerFixture {
    SchedulerFixture() : scheduler(SchedulerType::SYNCHRONOUS) {
        if (scheduler.create(nullptr) != Result::SUCCESS) {
            throw std::runtime_error("failed to create scheduler fixture");
        }
    }

    ~SchedulerFixture() {
        shutdown();
    }

    std::shared_ptr<Module> module(const std::string& name,
                                   const TensorMap& inputs = {},
                                   RuntimeType runtime = RuntimeType::NATIVE) {
        auto value = makeModule(name, inputs, runtime);
        modules.push_back(value);
        return value;
    }

    Result shutdown() {
        if (closed) {
            return Result::SUCCESS;
        }

        closed = true;
        Result result = scheduler.destroy();

        for (auto it = modules.rbegin(); it != modules.rend(); ++it) {
            const auto moduleResult = (*it)->destroy();
            if (result == Result::SUCCESS && moduleResult != Result::SUCCESS) {
                result = moduleResult;
            }
        }

        return result;
    }

    Scheduler scheduler;
    std::vector<std::shared_ptr<Module>> modules;
    bool closed = false;
};

}  // namespace

TEST_CASE("Scheduler executes CPU modules in topological order",
          "[core][runtime][scheduler]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("topology_source");
    state.configure("topology_middle").inputs = {"in"};
    state.configure("topology_sink").inputs = {"in"};

    SchedulerFixture fixture;
    auto source = fixture.module("topology_source");
    auto middle = fixture.module("topology_middle", makeInputs({{"in", "topology_source"}}));
    auto sink = fixture.module("topology_sink", makeInputs({{"in", "topology_middle"}}));

    REQUIRE(fixture.scheduler.add(sink) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.add(middle) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.add(source) == Result::SUCCESS);

    state.clearEvents();
    std::unordered_set<std::string> failedModules = {"stale"};
    REQUIRE(fixture.scheduler.compute(failedModules) == Result::SUCCESS);

    REQUIRE(failedModules.empty());
    REQUIRE(state.calls("compute") ==
             std::vector<std::string>{"topology_source", "topology_middle", "topology_sink"});
}

TEST_CASE("Scheduler preserves insertion order for independent CPU modules",
          "[core][runtime][scheduler]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("independent_first");
    state.configure("independent_second");
    state.configure("independent_third");

    SchedulerFixture fixture;
    auto first = fixture.module("independent_first");
    auto second = fixture.module("independent_second");
    auto third = fixture.module("independent_third");
    REQUIRE(fixture.scheduler.add(first) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.add(second) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.add(third) == Result::SUCCESS);

    state.clearEvents();
    std::unordered_set<std::string> failedModules;
    REQUIRE(fixture.scheduler.compute(failedModules) == Result::SUCCESS);
    REQUIRE(failedModules.empty());
    // Current defect: independent modules are ordered by unordered-map iteration, not insertion order.
    REQUIRE(state.calls("compute") == std::vector<std::string>{
                                          "independent_first",
                                          "independent_second",
                                          "independent_third",
                                      });
}

TEST_CASE("Scheduler treats a missing producer as a provisional CPU source",
          "[core][runtime][scheduler]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("late_consumer").inputs = {"in"};
    state.configure("late_producer");

    SchedulerFixture fixture;
    auto consumer = fixture.module("late_consumer", makeInputs({{"in", "late_producer"}}));
    auto producer = fixture.module("late_producer");
    REQUIRE(fixture.scheduler.add(consumer) == Result::SUCCESS);

    state.clearEvents();
    std::unordered_set<std::string> failedModules;
    REQUIRE(fixture.scheduler.compute(failedModules) == Result::SUCCESS);
    REQUIRE(failedModules.empty());
    REQUIRE(state.calls("compute") == std::vector<std::string>{"late_consumer"});

    REQUIRE(fixture.scheduler.add(producer) == Result::SUCCESS);
    state.clearEvents();
    REQUIRE(fixture.scheduler.compute(failedModules) == Result::SUCCESS);
    REQUIRE(failedModules.empty());
    REQUIRE(state.calls("compute") ==
            std::vector<std::string>{"late_producer", "late_consumer"});
}

TEST_CASE("Scheduler executes a CPU diamond once in dependency order",
          "[core][runtime][scheduler]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("diamond_source");
    state.configure("diamond_left").inputs = {"in"};
    state.configure("diamond_right").inputs = {"in"};
    state.configure("diamond_sink").inputs = {"left", "right"};

    SchedulerFixture fixture;
    auto sink = fixture.module("diamond_sink", makeInputs({
                                                   {"left", "diamond_left"},
                                                   {"right", "diamond_right"},
                                               }));
    auto right = fixture.module("diamond_right", makeInputs({{"in", "diamond_source"}}));
    auto left = fixture.module("diamond_left", makeInputs({{"in", "diamond_source"}}));
    auto source = fixture.module("diamond_source");
    REQUIRE(fixture.scheduler.add(sink) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.add(right) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.add(left) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.add(source) == Result::SUCCESS);

    state.clearEvents();
    std::unordered_set<std::string> failedModules;
    REQUIRE(fixture.scheduler.compute(failedModules) == Result::SUCCESS);
    const auto calls = state.calls("compute");
    REQUIRE(failedModules.empty());
    REQUIRE(calls.size() == 4);
    REQUIRE(calls.front() == "diamond_source");
    REQUIRE(calls.back() == "diamond_sink");
    REQUIRE(std::unordered_set<std::string>(calls.begin() + 1, calls.end() - 1) ==
            std::unordered_set<std::string>{"diamond_left", "diamond_right"});
}

TEST_CASE("Scheduler executes every CPU fan-out branch once",
          "[core][runtime][scheduler]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("fanout_source");
    state.configure("fanout_left").inputs = {"in"};
    state.configure("fanout_middle").inputs = {"in"};
    state.configure("fanout_right").inputs = {"in"};

    SchedulerFixture fixture;
    auto right = fixture.module("fanout_right", makeInputs({{"in", "fanout_source"}}));
    auto middle = fixture.module("fanout_middle", makeInputs({{"in", "fanout_source"}}));
    auto left = fixture.module("fanout_left", makeInputs({{"in", "fanout_source"}}));
    auto source = fixture.module("fanout_source");
    REQUIRE(fixture.scheduler.add(right) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.add(middle) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.add(left) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.add(source) == Result::SUCCESS);

    state.clearEvents();
    std::unordered_set<std::string> failedModules;
    REQUIRE(fixture.scheduler.compute(failedModules) == Result::SUCCESS);
    const auto calls = state.calls("compute");
    REQUIRE(failedModules.empty());
    REQUIRE(calls.size() == 4);
    REQUIRE(calls.front() == "fanout_source");
    REQUIRE(std::unordered_set<std::string>(calls.begin() + 1, calls.end()) ==
            std::unordered_set<std::string>{"fanout_left", "fanout_middle", "fanout_right"});
}

TEST_CASE("Scheduler propagates SKIP only through dependent modules",
          "[core][runtime][scheduler]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("skip_source").computeResults = {Result::SKIP};
    state.configure("skip_middle").inputs = {"in"};
    state.configure("skip_sink").inputs = {"in"};
    state.configure("skip_independent");

    SchedulerFixture fixture;
    auto source = fixture.module("skip_source");
    auto middle = fixture.module("skip_middle", makeInputs({{"in", "skip_source"}}));
    auto sink = fixture.module("skip_sink", makeInputs({{"in", "skip_middle"}}));
    auto independent = fixture.module("skip_independent");
    REQUIRE(fixture.scheduler.add(sink) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.add(middle) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.add(independent) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.add(source) == Result::SUCCESS);

    state.clearEvents();
    std::unordered_set<std::string> failedModules = {"stale"};
    REQUIRE(fixture.scheduler.compute(failedModules) == Result::SUCCESS);
    REQUIRE(failedModules.empty());
    REQUIRE(state.count("compute", "skip_source") == 1);
    REQUIRE(state.count("compute", "skip_middle") == 0);
    REQUIRE(state.count("compute", "skip_sink") == 0);
    REQUIRE(state.count("compute", "skip_independent") == 1);
}

TEST_CASE("Scheduler reports a runtime submission failure and stops its segment",
          "[core][runtime][scheduler]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("failure_source").computeResults = {Result::ERROR};
    state.configure("failure_sink").inputs = {"in"};
    state.configure("failure_later").inputs = {"in"};

    SchedulerFixture fixture;
    auto source = fixture.module("failure_source");
    auto sink = fixture.module("failure_sink", makeInputs({{"in", "failure_source"}}));
    auto later = fixture.module("failure_later", makeInputs({{"in", "failure_sink"}}));
    REQUIRE(fixture.scheduler.add(later) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.add(sink) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.add(source) == Result::SUCCESS);

    state.clearEvents();
    std::unordered_set<std::string> failedModules = {"stale"};
    REQUIRE(fixture.scheduler.compute(failedModules) == Result::ERROR);
    REQUIRE(failedModules == std::unordered_set<std::string>{"failure_source"});
    REQUIRE(state.calls("compute") == std::vector<std::string>{"failure_source"});
}

TEST_CASE("Scheduler bounds YIELD and TIMEOUT source polling",
          "[core][runtime][scheduler]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("polling_source").pendingComputeResults = {
        Result::YIELD,
        Result::TIMEOUT,
        Result::SUCCESS,
    };

    SchedulerFixture fixture;
    auto source = fixture.module("polling_source");
    REQUIRE(fixture.scheduler.add(source) == Result::SUCCESS);

    state.clearEvents();
    std::unordered_set<std::string> failedModules;
    REQUIRE(fixture.scheduler.compute(failedModules) == Result::SUCCESS);

    REQUIRE(failedModules.empty());
    REQUIRE(state.count("pending_compute", "polling_source") == 3);
    REQUIRE(state.count("compute", "polling_source") == 1);
}

TEST_CASE("Scheduler propagates source polling errors without submitting compute",
          "[core][runtime][scheduler]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("polling_error").pendingComputeResults = {Result::ERROR};

    SchedulerFixture fixture;
    auto source = fixture.module("polling_error");
    REQUIRE(fixture.scheduler.add(source) == Result::SUCCESS);

    state.clearEvents();
    std::unordered_set<std::string> failedModules = {"stale"};
    REQUIRE(fixture.scheduler.compute(failedModules) == Result::ERROR);
    REQUIRE(failedModules == std::unordered_set<std::string>{"polling_error"});
    REQUIRE(state.count("pending_compute", "polling_error") == 1);
    REQUIRE(state.count("compute", "polling_error") == 0);
}

TEST_CASE("Scheduler stop cancels source polling before compute submission",
          "[core][runtime][scheduler]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("cancel_source");

    SchedulerFixture fixture;
    auto source = fixture.module("cancel_source");
    REQUIRE(fixture.scheduler.add(source) == Result::SUCCESS);

    state.clearEvents();
    Result stopResult = Result::ERROR;
    state.configure("cancel_source").pendingAction = [&] {
        stopResult = fixture.scheduler.stop();
    };

    std::unordered_set<std::string> failedModules;
    const auto computeResult = fixture.scheduler.compute(failedModules);

    REQUIRE(stopResult == Result::SUCCESS);
    REQUIRE(computeResult == Result::SUCCESS);
    REQUIRE(failedModules.empty());
    REQUIRE(state.count("pending_compute", "cancel_source") == 1);
    REQUIRE(state.count("compute", "cancel_source") == 0);
}

TEST_CASE("Scheduler rolls back a failed CPU Runtime addition",
          "[core][runtime][scheduler]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("rollback_stable");
    auto& rejectedSpec = state.configure("rollback_rejected");
    rejectedSpec.inputs = {"in"};
    rejectedSpec.initializeResults = {Result::ERROR};

    SchedulerFixture fixture;
    auto stable = fixture.module("rollback_stable");
    auto rejected = fixture.module("rollback_rejected", makeInputs({{"in", "rollback_stable"}}));
    REQUIRE(fixture.scheduler.add(stable) == Result::SUCCESS);

    state.clearEvents();
    REQUIRE(fixture.scheduler.add(rejected) == Result::ERROR);
    REQUIRE(state.count("initialize", "rollback_rejected") == 1);
    REQUIRE(state.count("deinitialize", "rollback_rejected") == 1);
    REQUIRE(state.count("initialize", "rollback_stable") >= 1);
    REQUIRE(state.count("initialize", "rollback_stable") ==
            state.count("deinitialize", "rollback_stable"));

    state.clearEvents();
    std::unordered_set<std::string> failedModules;
    REQUIRE(fixture.scheduler.compute(failedModules) == Result::SUCCESS);
    REQUIRE(failedModules.empty());
    REQUIRE(state.count("compute", "rollback_stable") == 1);
    REQUIRE(state.count("compute", "rollback_rejected") == 0);
}

TEST_CASE("Scheduler rejects a cycle and restores the previous graph",
          "[core][runtime][scheduler]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("cycle_a").inputs = {"in"};
    state.configure("cycle_b").inputs = {"in"};

    SchedulerFixture fixture;
    auto moduleA = fixture.module("cycle_a", makeInputs({{"in", "cycle_b"}}));
    auto moduleB = fixture.module("cycle_b", makeInputs({{"in", "cycle_a"}}));
    REQUIRE(fixture.scheduler.add(moduleA) == Result::SUCCESS);

    state.clearEvents();
    REQUIRE(fixture.scheduler.add(moduleB) == Result::ERROR);
    REQUIRE(state.count("initialize", "cycle_b") == 0);

    state.clearEvents();
    std::unordered_set<std::string> failedModules;
    REQUIRE(fixture.scheduler.compute(failedModules) == Result::SUCCESS);
    REQUIRE(failedModules.empty());
    REQUIRE(state.count("compute", "cycle_a") == 1);
    REQUIRE(state.count("compute", "cycle_b") == 0);
}

TEST_CASE("Scheduler rejects a CPU self-cycle and remains usable",
          "[core][runtime][scheduler]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("self_cycle").inputs = {"in"};
    state.configure("self_cycle_recovery");

    SchedulerFixture fixture;
    auto selfCycle = fixture.module("self_cycle", makeInputs({{"in", "self_cycle"}}));
    auto recovery = fixture.module("self_cycle_recovery");
    REQUIRE(fixture.scheduler.add(selfCycle) == Result::ERROR);
    REQUIRE(state.count("initialize", "self_cycle") == 0);
    REQUIRE(fixture.scheduler.add(recovery) == Result::SUCCESS);

    state.clearEvents();
    std::unordered_set<std::string> failedModules;
    REQUIRE(fixture.scheduler.compute(failedModules) == Result::SUCCESS);
    REQUIRE(failedModules.empty());
    REQUIRE(state.calls("compute") == std::vector<std::string>{"self_cycle_recovery"});
}

TEST_CASE("Scheduler lifecycle operations preserve CPU Runtime usability",
          "[core][runtime][scheduler]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("lifecycle_node");

    SchedulerFixture fixture;
    auto node = fixture.module("lifecycle_node");
    REQUIRE(fixture.scheduler.add(node) == Result::SUCCESS);

    state.clearEvents();
    REQUIRE(fixture.scheduler.stop() == Result::SUCCESS);
    std::unordered_set<std::string> failedModules = {"stale"};
    REQUIRE(fixture.scheduler.compute(failedModules) == Result::SUCCESS);
    REQUIRE(failedModules.empty());
    REQUIRE(state.count("compute", "lifecycle_node") == 0);
    REQUIRE(state.count("initialize", "lifecycle_node") == 0);
    REQUIRE(state.count("deinitialize", "lifecycle_node") == 0);

    REQUIRE(fixture.scheduler.start() == Result::SUCCESS);
    REQUIRE(fixture.scheduler.compute(failedModules) == Result::SUCCESS);
    REQUIRE(state.count("compute", "lifecycle_node") == 1);

    state.clearEvents();
    U64 synchronizedCalls = 0;
    REQUIRE(fixture.scheduler.synchronize([&] {
        synchronizedCalls += 1;
        return Result::SUCCESS;
    }) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.synchronize({}) == Result::SUCCESS);
    REQUIRE(synchronizedCalls == 1);
    REQUIRE(state.count("initialize", "lifecycle_node") == 0);
    REQUIRE(state.count("deinitialize", "lifecycle_node") == 0);

    REQUIRE(fixture.scheduler.synchronize([&] {
        synchronizedCalls += 1;
        return Result::ERROR;
    }) == Result::ERROR);
    REQUIRE(synchronizedCalls == 2);
    REQUIRE(fixture.scheduler.compute(failedModules) == Result::SUCCESS);
    REQUIRE(state.count("compute", "lifecycle_node") == 1);

    state.clearEvents();
    REQUIRE(fixture.scheduler.reload(node) == Result::SUCCESS);
    REQUIRE(state.events == std::vector<std::string>{
                                "deinitialize:lifecycle_node",
                                "initialize:lifecycle_node",
                            });

    state.clearEvents();
    REQUIRE(fixture.scheduler.remove(node) == Result::SUCCESS);
    REQUIRE(state.events == std::vector<std::string>{"deinitialize:lifecycle_node"});
    REQUIRE(fixture.scheduler.remove(node) == Result::SUCCESS);
    REQUIRE(state.count("deinitialize", "lifecycle_node") == 1);

    state.clearEvents();
    REQUIRE(fixture.scheduler.compute(failedModules) == Result::SUCCESS);
    REQUIRE(state.count("compute", "lifecycle_node") == 0);
}

TEST_CASE("Scheduler presents only modules with surface taint",
          "[core][runtime][scheduler]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("present_compute_only");
    state.configure("present_surface").surface = true;

    SchedulerFixture fixture;
    auto computeOnly = fixture.module("present_compute_only");
    auto surface = fixture.module("present_surface");
    REQUIRE(fixture.scheduler.add(computeOnly) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.add(surface) == Result::SUCCESS);
    REQUIRE(state.count("present_initialize", "present_compute_only") == 0);
    REQUIRE(state.count("present_initialize", "present_surface") == 1);

    state.clearEvents();
    std::unordered_set<std::string> failedModules;
    REQUIRE(fixture.scheduler.present(failedModules) == Result::SUCCESS);
    REQUIRE(failedModules.empty());
    REQUIRE(state.count("present", "present_surface") == 1);
    REQUIRE(state.count("present", "present_compute_only") == 0);
}

TEST_CASE("Scheduler reports presentation initialization and submission failures",
          "[core][runtime][scheduler]") {
    auto& state = syntheticState();

    SECTION("initialization failure does not add the module") {
        state.reset();
        auto& spec = state.configure("present_init_failure");
        spec.surface = true;
        spec.presentInitializeResults = {Result::ERROR};

        SchedulerFixture fixture;
        auto surface = fixture.module("present_init_failure");
        REQUIRE(fixture.scheduler.add(surface) == Result::ERROR);
        REQUIRE(state.count("present_initialize", "present_init_failure") == 1);
        REQUIRE(state.count("initialize", "present_init_failure") == 0);

        state.clearEvents();
        std::unordered_set<std::string> failedModules;
        REQUIRE(fixture.scheduler.present(failedModules) == Result::SUCCESS);
        REQUIRE(failedModules.empty());
        REQUIRE(state.count("present", "present_init_failure") == 0);
    }

    SECTION("submission failure names the surface") {
        state.reset();
        auto& spec = state.configure("present_submit_failure");
        spec.surface = true;
        spec.presentResults = {Result::ERROR};

        SchedulerFixture fixture;
        auto surface = fixture.module("present_submit_failure");
        REQUIRE(fixture.scheduler.add(surface) == Result::SUCCESS);

        state.clearEvents();
        std::unordered_set<std::string> failedModules = {"stale"};
        REQUIRE(fixture.scheduler.present(failedModules) == Result::ERROR);
        REQUIRE(failedModules == std::unordered_set<std::string>{"present_submit_failure"});
        REQUIRE(state.count("present", "present_submit_failure") == 1);
    }

    SECTION("RELOAD is successful presentation progress") {
        state.reset();
        auto& spec = state.configure("present_reload");
        spec.surface = true;
        spec.presentResults = {Result::RELOAD};

        SchedulerFixture fixture;
        auto surface = fixture.module("present_reload");
        REQUIRE(fixture.scheduler.add(surface) == Result::SUCCESS);

        state.clearEvents();
        std::unordered_set<std::string> failedModules = {"stale"};
        REQUIRE(fixture.scheduler.present(failedModules) == Result::SUCCESS);
        REQUIRE(failedModules.empty());
        REQUIRE(state.count("present", "present_reload") == 1);
    }
}

TEST_CASE("Scheduler presentation follows add stop remove and re-add lifecycle",
          "[core][runtime][scheduler]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("present_lifecycle").surface = true;

    SchedulerFixture fixture;
    auto surface = fixture.module("present_lifecycle");

    state.clearEvents();
    REQUIRE(fixture.scheduler.add(surface) == Result::SUCCESS);
    REQUIRE(state.events == std::vector<std::string>{
                                "present_initialize:present_lifecycle",
                                "initialize:present_lifecycle",
                            });

    state.clearEvents();
    std::unordered_set<std::string> failedModules;
    REQUIRE(fixture.scheduler.present(failedModules) == Result::SUCCESS);
    REQUIRE(failedModules.empty());
    REQUIRE(state.events == std::vector<std::string>{"present:present_lifecycle"});

    state.clearEvents();
    REQUIRE(fixture.scheduler.stop() == Result::SUCCESS);
    REQUIRE(fixture.scheduler.present(failedModules) == Result::SUCCESS);
    REQUIRE(state.events.empty());
    REQUIRE(fixture.scheduler.start() == Result::SUCCESS);
    REQUIRE(fixture.scheduler.present(failedModules) == Result::SUCCESS);
    REQUIRE(state.events == std::vector<std::string>{"present:present_lifecycle"});

    state.clearEvents();
    REQUIRE(fixture.scheduler.remove(surface) == Result::SUCCESS);
    REQUIRE(state.events == std::vector<std::string>{"deinitialize:present_lifecycle"});
    state.clearEvents();
    REQUIRE(fixture.scheduler.present(failedModules) == Result::SUCCESS);
    REQUIRE(state.events.empty());

    REQUIRE(fixture.scheduler.add(surface) == Result::SUCCESS);
    REQUIRE(state.events == std::vector<std::string>{
                                "present_initialize:present_lifecycle",
                                "initialize:present_lifecycle",
                            });

    state.clearEvents();
    REQUIRE(fixture.shutdown() == Result::SUCCESS);
    REQUIRE(state.events == std::vector<std::string>{
                                "deinitialize:present_lifecycle",
                                "destroy:present_lifecycle",
                            });
}

TEST_CASE("Scheduler destroys CPU Runtime state before module destruction",
          "[core][runtime][scheduler]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("cleanup_success");

    SchedulerFixture fixture;
    auto module = fixture.module("cleanup_success");
    REQUIRE(fixture.scheduler.add(module) == Result::SUCCESS);

    state.clearEvents();
    REQUIRE(fixture.shutdown() == Result::SUCCESS);
    REQUIRE(state.events == std::vector<std::string>{
                                "deinitialize:cleanup_success",
                                "destroy:cleanup_success",
                            });
}

TEST_CASE("Scheduler propagates CPU Runtime destruction failures",
          "[core][runtime][scheduler]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("destroy_failure").deinitializeResults = {Result::ERROR};

    SchedulerFixture fixture;
    auto module = fixture.module("destroy_failure");
    REQUIRE(fixture.scheduler.add(module) == Result::SUCCESS);

    // Current defect: scheduler destruction discards Runtime::destroy() failures.
    REQUIRE(fixture.shutdown() == Result::ERROR);
}

TEST_CASE("Scheduler settles fully static branches without stopping mixed inputs",
          "[core][runtime][scheduler][static]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("static_left").taint = Module::Taint::STATIC_OUTPUT;
    state.configure("static_transform").taint = Module::Taint::STATELESS;
    state.configure("static_transform").inputs = {"in"};
    state.configure("static_right").taint = Module::Taint::STATIC_OUTPUT;
    state.configure("static_merge").taint = Module::Taint::STATELESS;
    state.configure("static_merge").inputs = {"left", "right"};
    state.configure("live_source");
    state.configure("mixed_merge").taint = Module::Taint::STATELESS;
    state.configure("mixed_merge").inputs = {"left", "right"};
    state.configure("mixed_sink").taint = Module::Taint::STATELESS;
    state.configure("mixed_sink").inputs = {"in"};

    SchedulerFixture fixture;
    auto staticLeft = fixture.module("static_left");
    auto staticTransform = fixture.module(
        "static_transform",
        makeInputs({{"in", "static_left"}}));
    auto staticRight = fixture.module("static_right");
    auto staticMerge = fixture.module(
        "static_merge",
        makeInputs({{"left", "static_transform"}, {"right", "static_right"}}));
    auto liveSource = fixture.module("live_source");
    auto mixedMerge = fixture.module(
        "mixed_merge",
        makeInputs({{"left", "static_merge"}, {"right", "live_source"}}));
    auto mixedSink = fixture.module(
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
    for (const auto& module : modules) {
        REQUIRE(fixture.scheduler.add(module) == Result::SUCCESS);
    }

    std::unordered_set<std::string> failedModules;
    for (U64 cycle = 0; cycle < 3; ++cycle) {
        REQUIRE(fixture.scheduler.compute(failedModules) == Result::SUCCESS);
        REQUIRE(failedModules.empty());
    }

    REQUIRE(state.count("compute", "static_left") == 1);
    REQUIRE(state.count("compute", "static_transform") == 1);
    REQUIRE(state.count("compute", "static_right") == 1);
    REQUIRE(state.count("compute", "static_merge") == 1);
    REQUIRE(state.count("compute", "live_source") == 3);
    REQUIRE(state.count("compute", "mixed_merge") == 3);
    REQUIRE(state.count("compute", "mixed_sink") == 3);
    REQUIRE(staticLeft->timing().computeTime == 0.0f);
    REQUIRE(staticTransform->timing().computeTime == 0.0f);
    REQUIRE(staticMerge->timing().computeTime == 0.0f);

    REQUIRE(fixture.shutdown() == Result::SUCCESS);
}

TEST_CASE("Scheduler retries static modules until output is usable",
          "[core][runtime][scheduler][static]") {
    for (const auto initialResult : {Result::SKIP, Result::YIELD, Result::TIMEOUT}) {
        DYNAMIC_SECTION("Initial result: " << initialResult) {
            auto& state = syntheticState();
            state.reset();
            state.configure("retry_static_source").taint = Module::Taint::STATIC_OUTPUT;
            state.configure("retry_static_source").computeResults = {
                initialResult,
                Result::SUCCESS,
            };
            state.configure("retry_static_sink").taint = Module::Taint::STATELESS;
            state.configure("retry_static_sink").inputs = {"in"};

            SchedulerFixture fixture;
            auto source = fixture.module("retry_static_source");
            auto sink = fixture.module(
                "retry_static_sink",
                makeInputs({{"in", "retry_static_source"}}));
            REQUIRE(fixture.scheduler.add(source) == Result::SUCCESS);
            REQUIRE(fixture.scheduler.add(sink) == Result::SUCCESS);

            std::unordered_set<std::string> failedModules;
            REQUIRE(fixture.scheduler.compute(failedModules) == Result::SUCCESS);
            REQUIRE(fixture.scheduler.compute(failedModules) == Result::SUCCESS);
            REQUIRE(fixture.scheduler.compute(failedModules) == Result::SUCCESS);
            REQUIRE(state.count("compute", "retry_static_source") == 2);
            REQUIRE(state.count("compute", "retry_static_sink") == 1);

            REQUIRE(fixture.shutdown() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Scheduler invalidates settlement only for state mutations",
          "[core][runtime][scheduler][static]") {
    auto& state = syntheticState();

    SECTION("stop and start preserve settlement while synchronize clears it") {
        state.reset();
        auto& sourceSpec = state.configure("lifecycle_static");
        sourceSpec.taint = Module::Taint::STATIC_OUTPUT;
        sourceSpec.computeDelay = std::chrono::milliseconds(1);

        SchedulerFixture fixture;
        auto source = fixture.module("lifecycle_static");
        REQUIRE(fixture.scheduler.add(source) == Result::SUCCESS);

        std::unordered_set<std::string> failedModules;
        REQUIRE(fixture.scheduler.compute(failedModules) == Result::SUCCESS);
        REQUIRE(fixture.scheduler.compute(failedModules) == Result::SUCCESS);
        REQUIRE(state.count("compute", "lifecycle_static") == 1);

        REQUIRE(fixture.scheduler.stop() == Result::SUCCESS);
        REQUIRE(fixture.scheduler.start() == Result::SUCCESS);
        REQUIRE(fixture.scheduler.compute(failedModules) == Result::SUCCESS);
        REQUIRE(state.count("compute", "lifecycle_static") == 1);

        REQUIRE(fixture.scheduler.synchronize([] { return Result::SUCCESS; }) == Result::SUCCESS);
        REQUIRE(fixture.scheduler.compute(failedModules) == Result::SUCCESS);
        REQUIRE(state.count("compute", "lifecycle_static") == 2);

        REQUIRE(fixture.shutdown() == Result::SUCCESS);
        REQUIRE(source->timing().computeTime > 0.0f);
    }

    SECTION("input mutation disables settlement for the whole graph") {
        state.reset();
        state.configure("in_place_static").taint = Module::Taint::STATIC_OUTPUT;
        state.configure("in_place_consumer").taint = Module::Taint::IN_PLACE;
        state.configure("in_place_consumer").inputs = {"in"};

        SchedulerFixture fixture;
        auto source = fixture.module("in_place_static");
        auto consumer = fixture.module(
            "in_place_consumer",
            makeInputs({{"in", "in_place_static"}}));
        REQUIRE(fixture.scheduler.add(source) == Result::SUCCESS);
        REQUIRE(fixture.scheduler.add(consumer) == Result::SUCCESS);

        std::unordered_set<std::string> failedModules;
        REQUIRE(fixture.scheduler.compute(failedModules) == Result::SUCCESS);
        REQUIRE(fixture.scheduler.compute(failedModules) == Result::SUCCESS);
        REQUIRE(state.count("compute", "in_place_static") == 2);
        REQUIRE(state.count("compute", "in_place_consumer") == 2);

        REQUIRE(fixture.shutdown() == Result::SUCCESS);
    }
}

TEST_CASE("Scheduler keeps settled surfaces presentable across compute cycles",
          "[core][runtime][scheduler][static]") {
    auto& state = syntheticState();
    state.reset();
    auto& surfaceSpec = state.configure("static_surface");
    surfaceSpec.surface = true;
    surfaceSpec.taint = Module::Taint::STATIC_OUTPUT | Module::Taint::THROTTLED;
    state.configure("static_surface_sink").inputs = {"in"};

    SchedulerFixture fixture;
    auto surface = fixture.module("static_surface");
    auto sink = fixture.module(
        "static_surface_sink",
        makeInputs({{"in", "static_surface"}}));
    REQUIRE(fixture.scheduler.add(surface) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.add(sink) == Result::SUCCESS);

    std::unordered_set<std::string> failedModules;
    REQUIRE(fixture.scheduler.compute(failedModules) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.compute(failedModules) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.present(failedModules) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.present(failedModules) == Result::SUCCESS);
    REQUIRE(state.count("pending_compute", "static_surface") == 1);
    REQUIRE(state.count("compute", "static_surface") == 1);
    REQUIRE(state.count("compute", "static_surface_sink") == 2);
    REQUIRE(state.count("present", "static_surface") == 2);

    REQUIRE(fixture.shutdown() == Result::SUCCESS);
}

TEST_CASE("Scheduler preserves settlement across runtime segments",
          "[core][runtime][scheduler][static]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("boundary_native_source").taint = Module::Taint::STATIC_OUTPUT;
    state.configure("boundary_native_source").computeResults = {
        Result::YIELD,
        Result::SUCCESS,
    };
    state.configure("boundary_python").taint = Module::Taint::STATELESS;
    state.configure("boundary_python").inputs = {"in"};
    state.configure("boundary_native_sink").inputs = {"in"};

    SchedulerFixture fixture;
    auto source = fixture.module("boundary_native_source");
    auto python = fixture.module(
        "boundary_python",
        makeInputs({{"in", "boundary_native_source"}}),
        RuntimeType::PYTHON);
    auto sink = fixture.module(
        "boundary_native_sink",
        makeInputs({{"in", "boundary_python"}}));
    REQUIRE(fixture.scheduler.add(source) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.add(python) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.add(sink) == Result::SUCCESS);

    std::unordered_set<std::string> failedModules;
    REQUIRE(fixture.scheduler.compute(failedModules) == Result::SUCCESS);
    REQUIRE(state.count("compute", "boundary_native_source") == 1);
    REQUIRE(state.count("compute", "boundary_python") == 0);
    REQUIRE(state.count("compute", "boundary_native_sink") == 0);
    REQUIRE(fixture.scheduler.compute(failedModules) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.compute(failedModules) == Result::SUCCESS);
    REQUIRE(state.count("compute", "boundary_native_source") == 2);
    REQUIRE(state.count("compute", "boundary_python") == 1);
    REQUIRE(state.count("compute", "boundary_native_sink") == 2);

    REQUIRE(fixture.shutdown() == Result::SUCCESS);
}
