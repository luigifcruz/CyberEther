#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <future>
#include <initializer_list>
#include <latch>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "jetstream/detail/module_impl.hh"
#include "jetstream/detail/scheduler_impl.hh"
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
    std::string aliasInput;
    std::string inheritInput;
    std::string deriveFrame;
    bool publishClone = false;
    std::string forwardInput;
    bool readsFrame = false;
    U64 integrate = 0;
    bool counter = false;
    bool skipping = false;
    bool alternating = false;
    bool surface = false;
    Module::Taint taint = Module::Taint::CLEAN;
    std::chrono::milliseconds computeDelay{0};
    std::function<void()> computeAction;
    std::vector<Result> initializeResults;
    std::vector<Result> deinitializeResults;
    std::vector<Result> pendingComputeResults;
    std::vector<Result> computeResults;
    std::vector<Result> presentInitializeResults;
    std::vector<Result> presentCaptureResults;
    std::vector<Result> presentResults;
    std::function<void()> pendingAction;
    std::function<void()> pollAction;
    std::function<Result()> pendingCompute;
};

struct SyntheticState {
    std::unordered_map<std::string, SyntheticSpec> specs;
    std::unordered_map<std::string, std::size_t> cursors;
    std::unordered_map<std::string, U64> counts;
    std::unordered_map<std::string, std::vector<std::vector<F32>>> reads;
    std::vector<std::string> events;

    void reset() {
        specs.clear();
        cursors.clear();
        counts.clear();
        reads.clear();
        events.clear();
    }

    void clearEvents() {
        counts.clear();
        reads.clear();
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
        const auto& spec = syntheticState().spec(name());
        if (!spec.aliasInput.empty()) {
            output = inputs().at(spec.aliasInput).tensor.clone();
        } else {
            JST_CHECK(output.create(DeviceType::CPU, DataType::F32, {1}));
            output.at<F32>(0) = spec.counter ? 0.0f : 1.0f;
            if (spec.counter) {
                JST_CHECK(output.setAttribute("frame", 0.0f));
            }
        }
        if (!spec.inheritInput.empty()) {
            JST_CHECK(output.propagateAttributes(inputs().at(spec.inheritInput).tensor));
        }
        if (!spec.deriveFrame.empty()) {
            const auto source = inputs().at(spec.deriveFrame).tensor;
            JST_CHECK(output.setDerivedAttribute("frame", [source] {
                return source.attribute("frame");
            }));
        }
        outputs()["out"].produced(name(), "out", spec.publishClone ? output.clone() : output);
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
        if (spec.computeAction) {
            spec.computeAction();
        }
        if (spec.computeDelay.count() > 0) {
            std::this_thread::sleep_for(spec.computeDelay);
        }

        auto& values = syntheticState().reads[name()].emplace_back();
        for (const auto& input : spec.inputs) {
            const auto& tensor = inputs().at(input).tensor;
            values.push_back(tensor.template at<F32>(0));
            if (spec.readsFrame && tensor.hasAttribute("frame")) {
                values.push_back(std::any_cast<F32>(tensor.attribute("frame")));
            }
        }

        if (spec.integrate > 0) {
            syntheticState().invoke("compute", name(), {});
            if (step == 0) {
                output.at<F32>(0) = 0.0f;
            }
            output.at<F32>(0) += inputs().at(spec.inputs.front()).tensor.template at<F32>(0);
            step = (step + 1) % spec.integrate;
            return step == 0 ? Result::SUCCESS : Result::SKIP;
        }

        if (spec.skipping) {
            syntheticState().invoke("compute", name(), {});
            return Result::SKIP;
        }

        if (spec.alternating) {
            syntheticState().invoke("compute", name(), {});
            stalling = !stalling;
            if (stalling) {
                return Result::YIELD;
            }
            if (!spec.forwardInput.empty()) {
                output.at<F32>(0) = inputs().at(spec.forwardInput).tensor.template at<F32>(0);
            }
            return Result::SUCCESS;
        }

        const auto result = syntheticState().invoke("compute", name(), spec.computeResults);
        if (result == Result::SUCCESS && spec.counter) {
            output.at<F32>(0) += 1.0f;
            JST_CHECK(output.setAttribute("frame", output.at<F32>(0)));
        }
        if (result == Result::SUCCESS && !spec.forwardInput.empty()) {
            output.at<F32>(0) = inputs().at(spec.forwardInput).tensor.template at<F32>(0);
        }
        return result;
    }

    Result presentInitialize() override {
        const auto& spec = syntheticState().spec(name());
        return syntheticState().invoke("present_initialize", name(), spec.presentInitializeResults);
    }

    Result presentCapture() override {
        const auto& spec = syntheticState().spec(name());
        return syntheticState().invoke("capture", name(), spec.presentCaptureResults);
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
        if (spec.pollAction) {
            auto action = std::move(spec.pollAction);
            spec.pollAction = {};
            action();
        }

        if (spec.pendingCompute) {
            syntheticState().invoke("pending_compute", name(), {});
            return spec.pendingCompute();
        }

        return syntheticState().invoke("pending_compute", name(), spec.pendingComputeResults);
    }

    Tensor output;
    U64 step = 0;
    bool stalling = false;
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

TensorMap linkOutputs(std::initializer_list<std::pair<std::string, std::shared_ptr<Module>>> links) {
    TensorMap inputs;

    for (const auto& [slot, producer] : links) {
        inputs[slot].produced(producer->name(), "out", producer->outputs().at("out").tensor);
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
    std::vector<std::string> submitted;
    for (const auto& name : {"topology_source", "topology_middle", "topology_sink"}) {
        state.configure(name).computeAction = [&, name] { submitted.emplace_back(name); };
    }
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
    REQUIRE(submitted == state.calls("compute"));
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

TEST_CASE("Scheduler polls YIELD and TIMEOUT sources once per cycle",
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
    REQUIRE(state.count("pending_compute", "polling_source") == 1);
    REQUIRE(state.count("compute", "polling_source") == 0);

    REQUIRE(fixture.scheduler.compute(failedModules) == Result::SUCCESS);
    REQUIRE(state.count("pending_compute", "polling_source") == 2);
    REQUIRE(state.count("compute", "polling_source") == 0);

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

TEST_CASE("Scheduler pause returns at once without polling sources until resume",
          "[core][runtime][scheduler][pause]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("paused_source");

    SchedulerFixture fixture;
    auto source = fixture.module("paused_source");
    REQUIRE(fixture.scheduler.add(source) == Result::SUCCESS);

    REQUIRE_FALSE(fixture.scheduler.paused());
    REQUIRE(fixture.scheduler.pause() == Result::SUCCESS);
    REQUIRE(fixture.scheduler.paused());

    state.clearEvents();
    std::unordered_set<std::string> failedModules;
    const auto begin = std::chrono::steady_clock::now();
    for (U64 call = 0; call < 200; ++call) {
        REQUIRE(fixture.scheduler.compute(failedModules) == Result::SUCCESS);
    }
    const auto elapsed = std::chrono::steady_clock::now() - begin;

    REQUIRE(failedModules.empty());
    REQUIRE(elapsed < std::chrono::milliseconds(100));
    REQUIRE(state.count("pending_compute", "paused_source") == 0);
    REQUIRE(state.count("compute", "paused_source") == 0);

    REQUIRE(fixture.scheduler.resume() == Result::SUCCESS);
    REQUIRE_FALSE(fixture.scheduler.paused());
    REQUIRE(fixture.scheduler.compute(failedModules) == Result::SUCCESS);
    REQUIRE(state.count("compute", "paused_source") == 1);
}

TEST_CASE("Scheduler resume restarts a compute thread polling while paused",
          "[core][runtime][scheduler][pause]") {
    auto& state = syntheticState();
    state.reset();
    std::atomic<bool> computed{false};
    state.configure("wake_source").computeAction = [&] {
        computed.store(true);
    };

    SchedulerFixture fixture;
    auto source = fixture.module("wake_source");
    REQUIRE(fixture.scheduler.add(source) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.pause() == Result::SUCCESS);

    std::atomic<U64> loops{0};
    struct ComputeWorker {
        std::atomic<bool> stopRequested{false};
        std::thread thread;

        ~ComputeWorker() {
            stopRequested.store(true);
            if (thread.joinable()) {
                thread.join();
            }
        }
    } worker;

    worker.thread = std::thread([&] {
        std::unordered_set<std::string> computeFailures;
        while (!worker.stopRequested.load()) {
            (void)fixture.scheduler.compute(computeFailures);
            loops.fetch_add(1);
            if (fixture.scheduler.paused()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    CHECK(loops.load() > 0);
    CHECK_FALSE(computed.load());

    REQUIRE(fixture.scheduler.resume() == Result::SUCCESS);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!computed.load() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    CHECK(computed.load());

    REQUIRE(fixture.scheduler.pause() == Result::SUCCESS);
    REQUIRE(fixture.scheduler.stop() == Result::SUCCESS);
}

TEST_CASE("Scheduler pause cancels source polling before compute submission",
          "[core][runtime][scheduler][pause]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("pause_polling_source");

    SchedulerFixture fixture;
    auto source = fixture.module("pause_polling_source");
    REQUIRE(fixture.scheduler.add(source) == Result::SUCCESS);

    state.clearEvents();
    state.configure("pause_polling_source").pendingAction = [&] {
        REQUIRE(fixture.scheduler.pause() == Result::SUCCESS);
    };

    std::unordered_set<std::string> failedModules;
    REQUIRE(fixture.scheduler.compute(failedModules) == Result::SUCCESS);
    REQUIRE(failedModules.empty());
    REQUIRE(state.count("pending_compute", "pause_polling_source") == 1);
    REQUIRE(state.count("compute", "pause_polling_source") == 0);
}

TEST_CASE("Scheduler pause stops source polling after a ready source",
          "[core][runtime][scheduler][pause]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("pause_ready_first");
    state.configure("pause_ready_second");

    SchedulerFixture fixture;
    auto first = fixture.module("pause_ready_first");
    auto second = fixture.module("pause_ready_second");
    REQUIRE(fixture.scheduler.add(first) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.add(second) == Result::SUCCESS);

    state.clearEvents();
    const auto pause = [&] {
        REQUIRE(fixture.scheduler.pause() == Result::SUCCESS);
    };
    state.configure("pause_ready_first").pollAction = pause;
    state.configure("pause_ready_second").pollAction = pause;

    std::unordered_set<std::string> failedModules;
    REQUIRE(fixture.scheduler.compute(failedModules) == Result::SUCCESS);
    REQUIRE(failedModules.empty());
    REQUIRE(state.calls("pending_compute").size() == 1);
    REQUIRE(state.calls("compute").empty());

    state.configure("pause_ready_first").pollAction = {};
    state.configure("pause_ready_second").pollAction = {};
    REQUIRE(fixture.scheduler.resume() == Result::SUCCESS);
    REQUIRE(fixture.scheduler.compute(failedModules) == Result::SUCCESS);
    REQUIRE(state.calls("compute").size() == 2);
}

TEST_CASE("Scheduler pause finishes a started cycle and holds the next one",
          "[core][runtime][scheduler][pause]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("pause_segment_source");
    state.configure("pause_segment_python").inputs = {"in"};
    state.configure("pause_segment_sink").inputs = {"in"};

    SchedulerFixture fixture;
    auto source = fixture.module("pause_segment_source");
    auto python = fixture.module(
        "pause_segment_python",
        makeInputs({{"in", "pause_segment_source"}}),
        RuntimeType::PYTHON);
    auto sink = fixture.module(
        "pause_segment_sink",
        makeInputs({{"in", "pause_segment_python"}}));
    REQUIRE(fixture.scheduler.add(source) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.add(python) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.add(sink) == Result::SUCCESS);

    state.clearEvents();
    state.configure("pause_segment_source").computeAction = [&] {
        REQUIRE(fixture.scheduler.pause() == Result::SUCCESS);
    };

    std::unordered_set<std::string> failedModules;
    REQUIRE(fixture.scheduler.compute(failedModules) == Result::SUCCESS);
    REQUIRE(failedModules.empty());
    REQUIRE(fixture.scheduler.paused());
    const std::vector<std::string> cycle = {
        "pause_segment_source",
        "pause_segment_python",
        "pause_segment_sink",
    };
    REQUIRE(state.calls("compute") == cycle);

    state.configure("pause_segment_source").computeAction = {};
    REQUIRE(fixture.scheduler.compute(failedModules) == Result::SUCCESS);
    REQUIRE(state.calls("compute") == cycle);

    REQUIRE(fixture.scheduler.resume() == Result::SUCCESS);
    REQUIRE(fixture.scheduler.compute(failedModules) == Result::SUCCESS);
    REQUIRE(state.calls("compute") == std::vector<std::string>{
        "pause_segment_source",
        "pause_segment_python",
        "pause_segment_sink",
        "pause_segment_source",
        "pause_segment_python",
        "pause_segment_sink",
    });

    REQUIRE(fixture.shutdown() == Result::SUCCESS);
}

TEST_CASE("Scheduler mutates and stops cleanly while paused",
          "[core][runtime][scheduler][pause]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("paused_mutation_source");
    state.configure("paused_mutation_sink").inputs = {"in"};

    SchedulerFixture fixture;
    auto source = fixture.module("paused_mutation_source");
    auto sink = fixture.module("paused_mutation_sink", makeInputs({{"in", "paused_mutation_source"}}));
    REQUIRE(fixture.scheduler.add(source) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.pause() == Result::SUCCESS);

    REQUIRE(fixture.scheduler.add(sink) == Result::SUCCESS);
    bool synchronized = false;
    REQUIRE(fixture.scheduler.synchronize([&] {
        synchronized = true;
        return Result::SUCCESS;
    }) == Result::SUCCESS);
    REQUIRE(synchronized);
    REQUIRE(fixture.scheduler.paused());

    state.clearEvents();
    std::unordered_set<std::string> failedModules;
    REQUIRE(fixture.scheduler.compute(failedModules) == Result::SUCCESS);
    REQUIRE(state.calls("compute").empty());

    REQUIRE(fixture.scheduler.stop() == Result::SUCCESS);
    REQUIRE(fixture.scheduler.compute(failedModules) == Result::SUCCESS);
    REQUIRE(state.calls("compute").empty());

    REQUIRE(fixture.scheduler.start() == Result::SUCCESS);
    REQUIRE(fixture.scheduler.paused());
    REQUIRE(fixture.scheduler.compute(failedModules) == Result::SUCCESS);
    REQUIRE(state.calls("compute").empty());

    REQUIRE(fixture.scheduler.resume() == Result::SUCCESS);
    REQUIRE(fixture.scheduler.compute(failedModules) == Result::SUCCESS);
    REQUIRE(state.calls("compute") ==
            std::vector<std::string>{"paused_mutation_source", "paused_mutation_sink"});

    REQUIRE(fixture.scheduler.pause() == Result::SUCCESS);
    REQUIRE(fixture.scheduler.stop() == Result::SUCCESS);
    REQUIRE(fixture.shutdown() == Result::SUCCESS);
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

TEST_CASE("Scheduler compute does not wait for a later frame after a skipped present",
          "[core][runtime][scheduler]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("handoff_source");
    state.configure("handoff_surface").surface = true;

    SchedulerFixture fixture;
    auto source = fixture.module("handoff_source");
    auto surface = fixture.module("handoff_surface");
    REQUIRE(fixture.scheduler.add(source) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.add(surface) == Result::SUCCESS);

    Result presentResult = Result::ERROR;
    std::chrono::steady_clock::duration presentTime{};
    state.configure("handoff_source").computeAction = [&] {
        std::thread presenter([&] {
            std::unordered_set<std::string> failedModules;
            const auto start = std::chrono::steady_clock::now();
            presentResult = fixture.scheduler.present(failedModules);
            presentTime = std::chrono::steady_clock::now() - start;
        });
        presenter.join();
    };

    state.clearEvents();
    std::unordered_set<std::string> failedModules;
    REQUIRE(fixture.scheduler.compute(failedModules) == Result::SUCCESS);
    state.configure("handoff_source").computeAction = {};

    REQUIRE(presentResult == Result::SUCCESS);
    REQUIRE(presentTime < std::chrono::milliseconds(50));
    REQUIRE(state.count("capture", "handoff_surface") == 0);

    auto next = std::async(std::launch::async, [&fixture] {
        std::unordered_set<std::string> nextFailedModules;
        return fixture.scheduler.compute(nextFailedModules);
    });
    const bool finished = next.wait_for(std::chrono::seconds(1)) == std::future_status::ready;
    if (!finished) {
        REQUIRE(fixture.scheduler.stop() == Result::SUCCESS);
    }

    REQUIRE(finished);
    REQUIRE(next.get() == Result::SUCCESS);
    REQUIRE(state.count("compute", "handoff_source") == 2);
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

    SECTION("capture failure names the surface and skips submission") {
        state.reset();
        auto& spec = state.configure("present_capture_failure");
        spec.surface = true;
        spec.presentCaptureResults = {Result::ERROR};

        SchedulerFixture fixture;
        auto surface = fixture.module("present_capture_failure");
        REQUIRE(fixture.scheduler.add(surface) == Result::SUCCESS);

        state.clearEvents();
        std::unordered_set<std::string> failedModules = {"stale"};
        REQUIRE(fixture.scheduler.present(failedModules) == Result::ERROR);
        REQUIRE(failedModules == std::unordered_set<std::string>{"present_capture_failure"});
        REQUIRE(state.count("capture", "present_capture_failure") == 1);
        REQUIRE(state.count("present", "present_capture_failure") == 0);
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

TEST_CASE("Scheduler submits presentation while compute holds the data lock",
          "[core][runtime][scheduler]") {
    auto& state = syntheticState();
    state.reset();
    std::latch entered(1);
    std::latch release(1);
    state.configure("busy_source").computeAction = [&] {
        entered.count_down();
        release.wait();
    };
    state.configure("live_surface").surface = true;

    SchedulerFixture fixture;
    auto source = fixture.module("busy_source");
    auto surface = fixture.module("live_surface");
    REQUIRE(fixture.scheduler.add(source) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.add(surface) == Result::SUCCESS);

    Result computeResult = Result::ERROR;
    std::thread computeThread([&] {
        std::unordered_set<std::string> computeFailures;
        computeResult = fixture.scheduler.compute(computeFailures);
    });
    entered.wait();

    state.clearEvents();
    std::unordered_set<std::string> failedModules;
    REQUIRE(fixture.scheduler.present(failedModules) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.present(failedModules) == Result::SUCCESS);
    REQUIRE(failedModules.empty());
    REQUIRE(state.count("capture", "live_surface") == 0);
    REQUIRE(state.count("present", "live_surface") == 2);

    release.count_down();
    computeThread.join();
    REQUIRE(computeResult == Result::SUCCESS);

    state.clearEvents();
    REQUIRE(fixture.scheduler.present(failedModules) == Result::SUCCESS);
    REQUIRE(state.events == std::vector<std::string>{
                                "capture:live_surface",
                                "present:live_surface",
                            });
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
    REQUIRE(state.events == std::vector<std::string>{
                                "capture:present_lifecycle",
                                "present:present_lifecycle",
                            });

    state.clearEvents();
    REQUIRE(fixture.scheduler.stop() == Result::SUCCESS);
    REQUIRE(fixture.scheduler.present(failedModules) == Result::SUCCESS);
    REQUIRE(state.events.empty());
    REQUIRE(fixture.scheduler.start() == Result::SUCCESS);
    REQUIRE(fixture.scheduler.present(failedModules) == Result::SUCCESS);
    REQUIRE(state.events == std::vector<std::string>{
                                "capture:present_lifecycle",
                                "present:present_lifecycle",
                            });

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

namespace {

void computeCycles(Scheduler& scheduler, U64 cycles) {
    std::unordered_set<std::string> failedModules;

    for (U64 cycle = 0; cycle < cycles; ++cycle) {
        REQUIRE(scheduler.compute(failedModules) == Result::SUCCESS);
        REQUIRE(failedModules.empty());
    }
}

void skipSchedulerTimingOnCI() {
    if (std::getenv("CI")) {
        SKIP("Scheduler wall-clock timing tests are disabled in CI.");
    }
}

std::chrono::milliseconds timedCompute(Scheduler& scheduler) {
    std::unordered_set<std::string> failedModules;
    const auto start = std::chrono::steady_clock::now();
    const auto epoch = Scheduler::Impl::BeginComputeRound();
    const auto result = scheduler.compute(failedModules);
    Scheduler::Impl::WaitComputeRound(epoch, std::chrono::milliseconds(100));
    const auto elapsed = std::chrono::steady_clock::now() - start;

    REQUIRE(result == Result::SUCCESS);
    REQUIRE(failedModules.empty());

    return std::chrono::duration_cast<std::chrono::milliseconds>(elapsed);
}

}  // namespace

TEST_CASE("Scheduler runs a ready branch while an unrelated source waits",
          "[core][runtime][scheduler][dataflow]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("idle_source").pendingCompute = [] { return Result::TIMEOUT; };
    state.configure("idle_sink").inputs = {"in"};
    state.configure("live_source");
    state.configure("live_sink").inputs = {"in"};

    SchedulerFixture fixture;
    auto idleSource = fixture.module("idle_source");
    auto idleSink = fixture.module("idle_sink", makeInputs({{"in", "idle_source"}}));
    auto liveSource = fixture.module("live_source");
    auto liveSink = fixture.module("live_sink", makeInputs({{"in", "live_source"}}));
    REQUIRE(fixture.scheduler.add(idleSource) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.add(idleSink) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.add(liveSource) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.add(liveSink) == Result::SUCCESS);

    state.clearEvents();
    computeCycles(fixture.scheduler, 3);

    REQUIRE(state.count("pending_compute", "idle_source") == 3);
    REQUIRE(state.count("compute", "idle_source") == 0);
    REQUIRE(state.count("compute", "idle_sink") == 0);
    REQUIRE(state.count("compute", "live_source") == 3);
    REQUIRE(state.count("compute", "live_sink") == 3);

    REQUIRE(fixture.shutdown() == Result::SUCCESS);
}

TEST_CASE("Scheduler feeds a waiting source from a sibling branch in the same graph",
          "[core][runtime][scheduler][dataflow]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("loop_receiver").pendingCompute = [&state] {
        return state.count("compute", "loop_transmitter") > 0 ? Result::SUCCESS : Result::TIMEOUT;
    };
    state.configure("loop_display").inputs = {"in"};
    state.configure("loop_radio");
    state.configure("loop_transmitter").inputs = {"in"};

    SchedulerFixture fixture;
    auto receiver = fixture.module("loop_receiver");
    auto display = fixture.module("loop_display", makeInputs({{"in", "loop_receiver"}}));
    auto radio = fixture.module("loop_radio");
    auto transmitter = fixture.module("loop_transmitter", makeInputs({{"in", "loop_radio"}}));
    REQUIRE(fixture.scheduler.add(receiver) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.add(display) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.add(radio) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.add(transmitter) == Result::SUCCESS);

    state.clearEvents();
    computeCycles(fixture.scheduler, 1);
    REQUIRE(state.count("compute", "loop_radio") == 1);
    REQUIRE(state.count("compute", "loop_transmitter") == 1);
    REQUIRE(state.count("compute", "loop_receiver") == 0);
    REQUIRE(state.count("compute", "loop_display") == 0);

    computeCycles(fixture.scheduler, 2);
    REQUIRE(state.count("compute", "loop_radio") == 3);
    REQUIRE(state.count("compute", "loop_transmitter") == 3);
    REQUIRE(state.count("compute", "loop_receiver") == 2);
    REQUIRE(state.count("compute", "loop_display") == 2);

    REQUIRE(fixture.shutdown() == Result::SUCCESS);
}

TEST_CASE("Scheduler runs a join every cycle when all inputs are ready",
          "[core][runtime][scheduler][dataflow]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("ready_left");
    state.configure("ready_right");
    state.configure("ready_join").inputs = {"left", "right"};
    state.configure("ready_sink").inputs = {"in"};

    SchedulerFixture fixture;
    auto left = fixture.module("ready_left");
    auto right = fixture.module("ready_right");
    auto join = fixture.module("ready_join", makeInputs({{"left", "ready_left"}, {"right", "ready_right"}}));
    auto sink = fixture.module("ready_sink", makeInputs({{"in", "ready_join"}}));
    REQUIRE(fixture.scheduler.add(sink) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.add(join) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.add(right) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.add(left) == Result::SUCCESS);

    state.clearEvents();
    computeCycles(fixture.scheduler, 3);

    REQUIRE(state.count("compute", "ready_left") == 3);
    REQUIRE(state.count("compute", "ready_right") == 3);
    REQUIRE(state.count("compute", "ready_join") == 3);
    REQUIRE(state.count("compute", "ready_sink") == 3);

    REQUIRE(fixture.shutdown() == Result::SUCCESS);
}

TEST_CASE("Scheduler holds a ready input at a join until the other input arrives",
          "[core][runtime][scheduler][dataflow]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("hold_ready");
    state.configure("hold_late").pendingComputeResults = {Result::TIMEOUT, Result::TIMEOUT};
    state.configure("hold_join").inputs = {"left", "right"};
    state.configure("hold_ticker");

    SchedulerFixture fixture;
    auto ready = fixture.module("hold_ready");
    auto late = fixture.module("hold_late");
    auto join = fixture.module("hold_join", makeInputs({{"left", "hold_ready"}, {"right", "hold_late"}}));
    auto ticker = fixture.module("hold_ticker");
    REQUIRE(fixture.scheduler.add(ready) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.add(late) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.add(join) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.add(ticker) == Result::SUCCESS);

    state.clearEvents();
    computeCycles(fixture.scheduler, 2);
    REQUIRE(state.count("compute", "hold_ready") == 1);
    REQUIRE(state.count("compute", "hold_late") == 0);
    REQUIRE(state.count("compute", "hold_join") == 0);

    computeCycles(fixture.scheduler, 1);
    REQUIRE(state.count("compute", "hold_ready") == 1);
    REQUIRE(state.count("compute", "hold_late") == 1);
    REQUIRE(state.count("compute", "hold_join") == 1);

    computeCycles(fixture.scheduler, 1);
    REQUIRE(state.count("compute", "hold_ready") == 2);
    REQUIRE(state.count("compute", "hold_late") == 2);
    REQUIRE(state.count("compute", "hold_join") == 2);
    REQUIRE(state.count("compute", "hold_ticker") == 4);

    REQUIRE(fixture.shutdown() == Result::SUCCESS);
}

TEST_CASE("Scheduler drops the sibling frame when one join input skips",
          "[core][runtime][scheduler][dataflow]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("sibling_skip").computeResults = {Result::SKIP};
    state.configure("sibling_live");
    state.configure("sibling_join").inputs = {"left", "right"};

    SchedulerFixture fixture;
    auto skip = fixture.module("sibling_skip");
    auto live = fixture.module("sibling_live");
    auto join = fixture.module("sibling_join", makeInputs({{"left", "sibling_skip"}, {"right", "sibling_live"}}));
    REQUIRE(fixture.scheduler.add(skip) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.add(live) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.add(join) == Result::SUCCESS);

    state.clearEvents();
    computeCycles(fixture.scheduler, 1);
    REQUIRE(state.count("compute", "sibling_skip") == 1);
    REQUIRE(state.count("compute", "sibling_live") == 1);
    REQUIRE(state.count("compute", "sibling_join") == 0);

    computeCycles(fixture.scheduler, 1);
    REQUIRE(state.count("compute", "sibling_skip") == 2);
    REQUIRE(state.count("compute", "sibling_live") == 2);
    REQUIRE(state.count("compute", "sibling_join") == 1);

    computeCycles(fixture.scheduler, 1);
    REQUIRE(state.count("compute", "sibling_skip") == 3);
    REQUIRE(state.count("compute", "sibling_live") == 3);
    REQUIRE(state.count("compute", "sibling_join") == 2);

    REQUIRE(fixture.shutdown() == Result::SUCCESS);
}

TEST_CASE("Scheduler paces a fork by its slowest consumer",
          "[core][runtime][scheduler][dataflow]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("fork_source");
    state.configure("fork_fast").inputs = {"in"};
    state.configure("fork_slow").inputs = {"in"};
    state.configure("fork_slow").computeResults = {Result::YIELD, Result::YIELD};
    state.configure("fork_ticker");

    SchedulerFixture fixture;
    auto source = fixture.module("fork_source");
    auto fast = fixture.module("fork_fast", makeInputs({{"in", "fork_source"}}));
    auto slow = fixture.module("fork_slow", makeInputs({{"in", "fork_source"}}));
    auto ticker = fixture.module("fork_ticker");
    REQUIRE(fixture.scheduler.add(source) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.add(fast) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.add(slow) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.add(ticker) == Result::SUCCESS);

    state.clearEvents();
    computeCycles(fixture.scheduler, 3);
    REQUIRE(state.count("compute", "fork_source") == 1);
    REQUIRE(state.count("compute", "fork_fast") == 1);
    REQUIRE(state.count("compute", "fork_slow") == 3);

    computeCycles(fixture.scheduler, 1);
    REQUIRE(state.count("compute", "fork_source") == 2);
    REQUIRE(state.count("compute", "fork_fast") == 2);
    REQUIRE(state.count("compute", "fork_slow") == 4);

    REQUIRE(fixture.shutdown() == Result::SUCCESS);
}

TEST_CASE("Scheduler resumes a stalled consumer on its held input",
          "[core][runtime][scheduler][dataflow]") {
    for (const auto status : {Result::YIELD, Result::TIMEOUT}) {
        DYNAMIC_SECTION("Stall result: " << status) {
            auto& state = syntheticState();
            state.reset();
            state.configure("resume_source");
            state.configure("resume_sink").inputs = {"in"};
            state.configure("resume_sink").computeResults = {status};
            state.configure("resume_ticker");

            SchedulerFixture fixture;
            auto source = fixture.module("resume_source");
            auto sink = fixture.module("resume_sink", makeInputs({{"in", "resume_source"}}));
            auto ticker = fixture.module("resume_ticker");
            REQUIRE(fixture.scheduler.add(source) == Result::SUCCESS);
            REQUIRE(fixture.scheduler.add(sink) == Result::SUCCESS);
            REQUIRE(fixture.scheduler.add(ticker) == Result::SUCCESS);

            state.clearEvents();
            computeCycles(fixture.scheduler, 1);
            REQUIRE(state.count("compute", "resume_source") == 1);
            REQUIRE(state.count("compute", "resume_sink") == 1);

            computeCycles(fixture.scheduler, 1);
            REQUIRE(state.count("compute", "resume_source") == 1);
            REQUIRE(state.count("compute", "resume_sink") == 2);

            computeCycles(fixture.scheduler, 1);
            REQUIRE(state.count("compute", "resume_source") == 2);
            REQUIRE(state.count("compute", "resume_sink") == 3);

            REQUIRE(fixture.shutdown() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Scheduler confines a stalled module to its branch within a segment",
          "[core][runtime][scheduler][dataflow]") {
    for (const auto runtime : {RuntimeType::NATIVE, RuntimeType::PYTHON}) {
        for (const auto status : {Result::YIELD, Result::TIMEOUT}) {
            DYNAMIC_SECTION("Runtime: " << runtime << ", stall result: " << status) {
                auto& state = syntheticState();
                state.reset();
                state.configure("confine_stall").computeResults = {status};
                state.configure("confine_stall_sink").inputs = {"in"};
                state.configure("confine_free");
                state.configure("confine_free_sink").inputs = {"in"};

                SchedulerFixture fixture;
                auto stall = fixture.module("confine_stall", {}, runtime);
                auto stallSink = fixture.module("confine_stall_sink",
                                                makeInputs({{"in", "confine_stall"}}),
                                                runtime);
                auto free = fixture.module("confine_free", {}, runtime);
                auto freeSink = fixture.module("confine_free_sink",
                                               makeInputs({{"in", "confine_free"}}),
                                               runtime);
                REQUIRE(fixture.scheduler.add(stall) == Result::SUCCESS);
                REQUIRE(fixture.scheduler.add(stallSink) == Result::SUCCESS);
                REQUIRE(fixture.scheduler.add(free) == Result::SUCCESS);
                REQUIRE(fixture.scheduler.add(freeSink) == Result::SUCCESS);

                state.clearEvents();
                computeCycles(fixture.scheduler, 1);
                REQUIRE(state.count("compute", "confine_stall") == 1);
                REQUIRE(state.count("compute", "confine_stall_sink") == 0);
                REQUIRE(state.count("compute", "confine_free") == 1);
                REQUIRE(state.count("compute", "confine_free_sink") == 1);

                computeCycles(fixture.scheduler, 1);
                REQUIRE(state.count("compute", "confine_stall") == 2);
                REQUIRE(state.count("compute", "confine_stall_sink") == 1);
                REQUIRE(state.count("compute", "confine_free") == 2);
                REQUIRE(state.count("compute", "confine_free_sink") == 2);

                REQUIRE(fixture.shutdown() == Result::SUCCESS);
            }
        }
    }
}

TEST_CASE("Scheduler moves fresh data through every runtime segment in one cycle",
          "[core][runtime][scheduler][dataflow]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("chain_source");
    state.configure("chain_python").inputs = {"in"};
    state.configure("chain_sink").inputs = {"in"};

    SchedulerFixture fixture;
    auto source = fixture.module("chain_source");
    auto python = fixture.module("chain_python", makeInputs({{"in", "chain_source"}}), RuntimeType::PYTHON);
    auto sink = fixture.module("chain_sink", makeInputs({{"in", "chain_python"}}));
    REQUIRE(fixture.scheduler.add(source) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.add(python) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.add(sink) == Result::SUCCESS);

    state.clearEvents();
    computeCycles(fixture.scheduler, 1);
    REQUIRE(state.calls("compute") == std::vector<std::string>{"chain_source", "chain_python", "chain_sink"});

    computeCycles(fixture.scheduler, 1);
    REQUIRE(state.count("compute", "chain_source") == 2);
    REQUIRE(state.count("compute", "chain_python") == 2);
    REQUIRE(state.count("compute", "chain_sink") == 2);

    REQUIRE(fixture.shutdown() == Result::SUCCESS);
}

TEST_CASE("Scheduler drops the sibling frame across runtime segments when one input skips",
          "[core][runtime][scheduler][dataflow]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("segment_skip").computeResults = {Result::SKIP};
    state.configure("segment_live");
    state.configure("segment_join").inputs = {"left", "right"};
    state.configure("segment_sink").inputs = {"in"};

    SchedulerFixture fixture;
    auto skip = fixture.module("segment_skip");
    auto live = fixture.module("segment_live");
    auto join = fixture.module("segment_join",
                               makeInputs({{"left", "segment_skip"}, {"right", "segment_live"}}),
                               RuntimeType::PYTHON);
    auto sink = fixture.module("segment_sink", makeInputs({{"in", "segment_join"}}));
    REQUIRE(fixture.scheduler.add(skip) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.add(live) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.add(join) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.add(sink) == Result::SUCCESS);

    state.clearEvents();
    computeCycles(fixture.scheduler, 1);
    REQUIRE(state.count("compute", "segment_skip") == 1);
    REQUIRE(state.count("compute", "segment_live") == 1);
    REQUIRE(state.count("compute", "segment_join") == 0);
    REQUIRE(state.count("compute", "segment_sink") == 0);

    computeCycles(fixture.scheduler, 1);
    REQUIRE(state.count("compute", "segment_skip") == 2);
    REQUIRE(state.count("compute", "segment_live") == 2);
    REQUIRE(state.count("compute", "segment_join") == 1);
    REQUIRE(state.count("compute", "segment_sink") == 1);

    computeCycles(fixture.scheduler, 1);
    REQUIRE(state.count("compute", "segment_skip") == 3);
    REQUIRE(state.count("compute", "segment_live") == 3);
    REQUIRE(state.count("compute", "segment_join") == 2);
    REQUIRE(state.count("compute", "segment_sink") == 2);

    REQUIRE(fixture.shutdown() == Result::SUCCESS);
}

TEST_CASE("Scheduler releases a held producer when the graph changes",
          "[core][runtime][scheduler][dataflow]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("release_source");
    state.configure("release_stuck").inputs = {"in"};
    state.configure("release_stuck").computeResults = std::vector<Result>(16, Result::YIELD);
    state.configure("release_reader").inputs = {"in"};
    state.configure("release_ticker");

    SchedulerFixture fixture;
    auto source = fixture.module("release_source");
    auto stuck = fixture.module("release_stuck", makeInputs({{"in", "release_source"}}));
    auto reader = fixture.module("release_reader", makeInputs({{"in", "release_source"}}));
    auto ticker = fixture.module("release_ticker");
    REQUIRE(fixture.scheduler.add(source) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.add(stuck) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.add(ticker) == Result::SUCCESS);

    state.clearEvents();
    computeCycles(fixture.scheduler, 3);
    REQUIRE(state.count("compute", "release_source") == 1);
    REQUIRE(state.count("compute", "release_stuck") == 3);

    REQUIRE(fixture.scheduler.remove(stuck) == Result::SUCCESS);
    computeCycles(fixture.scheduler, 2);
    REQUIRE(state.count("compute", "release_source") == 3);
    REQUIRE(state.count("compute", "release_stuck") == 3);

    REQUIRE(fixture.scheduler.add(reader) == Result::SUCCESS);
    computeCycles(fixture.scheduler, 2);
    REQUIRE(state.count("compute", "release_source") == 5);
    REQUIRE(state.count("compute", "release_reader") == 2);

    REQUIRE(fixture.shutdown() == Result::SUCCESS);
}

TEST_CASE("Scheduler keeps the producer of a throttled consumer flowing",
          "[core][runtime][scheduler][dataflow][throttle]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("throttle_source");
    state.configure("throttle_consumer").inputs = {"in"};
    state.configure("throttle_consumer").taint = Module::Taint::THROTTLED;

    SchedulerFixture fixture;
    auto source = fixture.module("throttle_source");
    auto consumer = fixture.module("throttle_consumer", makeInputs({{"in", "throttle_source"}}));
    REQUIRE(fixture.scheduler.add(source) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.add(consumer) == Result::SUCCESS);

    state.clearEvents();
    computeCycles(fixture.scheduler, 3);
    REQUIRE(state.count("compute", "throttle_source") == 3);
    REQUIRE(state.count("compute", "throttle_consumer") == 1);

    std::this_thread::sleep_for(std::chrono::milliseconds(120));

    computeCycles(fixture.scheduler, 1);
    REQUIRE(state.count("compute", "throttle_source") == 4);
    REQUIRE(state.count("compute", "throttle_consumer") == 2);

    computeCycles(fixture.scheduler, 1);
    REQUIRE(state.count("compute", "throttle_source") == 5);
    REQUIRE(state.count("compute", "throttle_consumer") == 2);

    REQUIRE(fixture.shutdown() == Result::SUCCESS);
}

TEST_CASE("Scheduler sleeps until the next throttle deadline on an idle graph",
          "[core][runtime][scheduler][dataflow][throttle]") {
    skipSchedulerTimingOnCI();
    auto& state = syntheticState();
    state.reset();
    state.configure("deadline_module").taint = Module::Taint::THROTTLED;

    SchedulerFixture fixture;
    auto module = fixture.module("deadline_module");
    REQUIRE(fixture.scheduler.add(module) == Result::SUCCESS);

    state.clearEvents();
    computeCycles(fixture.scheduler, 1);
    REQUIRE(state.count("compute", "deadline_module") == 1);

    REQUIRE(timedCompute(fixture.scheduler) >= std::chrono::milliseconds(50));
    REQUIRE(state.count("compute", "deadline_module") == 1);

    computeCycles(fixture.scheduler, 1);
    REQUIRE(state.count("compute", "deadline_module") == 2);

    REQUIRE(fixture.shutdown() == Result::SUCCESS);
}

TEST_CASE("Scheduler idle cycles wait for a pending compute signal",
          "[core][runtime][scheduler][dataflow][wake]") {
    skipSchedulerTimingOnCI();
    auto& state = syntheticState();
    state.reset();
    state.configure("wake_source").pendingCompute = [] { return Result::TIMEOUT; };

    SchedulerFixture fixture;
    auto source = fixture.module("wake_source");
    REQUIRE(fixture.scheduler.add(source) == Result::SUCCESS);

    SECTION("an idle cycle sleeps instead of spinning") {
        REQUIRE(timedCompute(fixture.scheduler) >= std::chrono::milliseconds(50));
        REQUIRE(state.count("compute", "wake_source") == 0);
    }

    SECTION("a signal from another thread ends the wait") {
        std::atomic<bool> done{false};
        std::thread signaller([&done] {
            while (!done.load()) {
                Scheduler::Context::NotifyPendingCompute();
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });

        const auto elapsed = timedCompute(fixture.scheduler);
        done.store(true);
        signaller.join();

        REQUIRE(elapsed < std::chrono::milliseconds(90));
    }

    SECTION("a signal raised while polling is not lost") {
        state.configure("wake_source").pendingCompute = [] {
            Scheduler::Context::NotifyPendingCompute();
            return Result::TIMEOUT;
        };

        REQUIRE(timedCompute(fixture.scheduler) < std::chrono::milliseconds(90));
    }

    SECTION("stop ends the wait") {
        std::thread stopper([&fixture] {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            REQUIRE(fixture.scheduler.stop() == Result::SUCCESS);
        });

        const auto elapsed = timedCompute(fixture.scheduler);
        stopper.join();

        REQUIRE(elapsed < std::chrono::milliseconds(90));
    }

    REQUIRE(fixture.shutdown() == Result::SUCCESS);
}

TEST_CASE("Scheduler does not wait after a cycle that made progress",
          "[core][runtime][scheduler][dataflow][wake]") {
    skipSchedulerTimingOnCI();
    auto& state = syntheticState();
    state.reset();
    state.configure("progress_source");
    state.configure("progress_idle").pendingCompute = [] { return Result::TIMEOUT; };

    SchedulerFixture fixture;
    auto source = fixture.module("progress_source");
    auto idle = fixture.module("progress_idle");
    REQUIRE(fixture.scheduler.add(source) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.add(idle) == Result::SUCCESS);

    for (U64 cycle = 0; cycle < 5; ++cycle) {
        REQUIRE(timedCompute(fixture.scheduler) < std::chrono::milliseconds(50));
    }
    REQUIRE(state.count("compute", "progress_source") == 5);

    REQUIRE(fixture.shutdown() == Result::SUCCESS);
}

TEST_CASE("Scheduler holds a producer while a view of its output waits downstream",
          "[core][runtime][scheduler][dataflow]") {
    for (const U64 views : {1, 2}) {
        DYNAMIC_SECTION("Views in chain: " << views) {
            auto& state = syntheticState();
            state.reset();
            state.configure("view_source");
            state.configure("view_late").pendingComputeResults = {Result::TIMEOUT, Result::TIMEOUT};
            state.configure("view_ticker");

            SchedulerFixture fixture;
            auto source = fixture.module("view_source");
            auto late = fixture.module("view_late");
            auto ticker = fixture.module("view_ticker");
            REQUIRE(fixture.scheduler.add(source) == Result::SUCCESS);
            REQUIRE(fixture.scheduler.add(late) == Result::SUCCESS);
            REQUIRE(fixture.scheduler.add(ticker) == Result::SUCCESS);

            std::string upstream = "view_source";
            for (U64 index = 0; index < views; ++index) {
                const std::string name = "view_" + std::to_string(index);
                state.configure(name).inputs = {"in"};
                state.configure(name).aliasInput = "in";
                auto view = fixture.module(name, makeInputs({{"in", upstream}}));
                REQUIRE(view->outputs().at("out").tensor.sharesStorage(view->inputs().at("in").tensor));
                REQUIRE(fixture.scheduler.add(view) == Result::SUCCESS);
                upstream = name;
            }

            state.configure("view_join").inputs = {"left", "right"};
            auto join = fixture.module("view_join", makeInputs({{"left", upstream}, {"right", "view_late"}}));
            REQUIRE(fixture.scheduler.add(join) == Result::SUCCESS);

            state.clearEvents();
            computeCycles(fixture.scheduler, 2);
            REQUIRE(state.count("compute", "view_source") == 1);
            REQUIRE(state.count("compute", upstream) == 1);
            REQUIRE(state.count("compute", "view_join") == 0);

            computeCycles(fixture.scheduler, 1);
            REQUIRE(state.count("compute", "view_source") == 1);
            REQUIRE(state.count("compute", "view_late") == 1);
            REQUIRE(state.count("compute", "view_join") == 1);

            computeCycles(fixture.scheduler, 1);
            REQUIRE(state.count("compute", "view_source") == 2);
            REQUIRE(state.count("compute", upstream) == 2);
            REQUIRE(state.count("compute", "view_join") == 2);

            REQUIRE(fixture.shutdown() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Scheduler decides throttle deadlines once per cycle across runtime segments",
          "[core][runtime][scheduler][dataflow][throttle]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("deadline_source");
    state.configure("deadline_view").inputs = {"in"};
    state.configure("deadline_view").aliasInput = "in";
    state.configure("deadline_display").inputs = {"in"};
    state.configure("deadline_display").taint = Module::Taint::THROTTLED;

    SchedulerFixture fixture;
    auto source = fixture.module("deadline_source", {}, RuntimeType::PYTHON);
    auto view = fixture.module("deadline_view", makeInputs({{"in", "deadline_source"}}));
    auto display = fixture.module("deadline_display", makeInputs({{"in", "deadline_view"}}));
    REQUIRE(fixture.scheduler.add(source) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.add(view) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.add(display) == Result::SUCCESS);

    state.clearEvents();
    computeCycles(fixture.scheduler, 2);
    REQUIRE(state.count("compute", "deadline_source") == 2);
    REQUIRE(state.count("compute", "deadline_view") == 2);
    REQUIRE(state.count("compute", "deadline_display") == 1);

    // Cross the 100 ms deadline inside compute, without a preparatory sleep
    // that could overshoot and make the display due before this cycle starts.
    state.configure("deadline_source").computeDelay = std::chrono::milliseconds(120);
    computeCycles(fixture.scheduler, 1);
    state.configure("deadline_source").computeDelay = std::chrono::milliseconds(0);
    REQUIRE(state.count("compute", "deadline_source") == 3);
    REQUIRE(state.count("compute", "deadline_view") == 3);
    REQUIRE(state.count("compute", "deadline_display") == 1);

    computeCycles(fixture.scheduler, 1);
    REQUIRE(state.count("compute", "deadline_source") == 4);
    REQUIRE(state.count("compute", "deadline_view") == 4);
    REQUIRE(state.count("compute", "deadline_display") == 2);

    computeCycles(fixture.scheduler, 1);
    REQUIRE(state.count("compute", "deadline_source") == 5);
    REQUIRE(state.count("compute", "deadline_view") == 5);
    REQUIRE(state.count("compute", "deadline_display") == 2);

    REQUIRE(fixture.shutdown() == Result::SUCCESS);
}

TEST_CASE("Scheduler does not sleep past a throttle deadline that expires during a cycle",
          "[core][runtime][scheduler][dataflow][throttle]") {
    skipSchedulerTimingOnCI();
    auto& state = syntheticState();
    state.reset();
    state.configure("expire_display").taint = Module::Taint::THROTTLED;
    state.configure("expire_slow");

    SchedulerFixture fixture;
    auto display = fixture.module("expire_display");
    auto slow = fixture.module("expire_slow");
    REQUIRE(fixture.scheduler.add(display) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.add(slow) == Result::SUCCESS);

    state.clearEvents();
    computeCycles(fixture.scheduler, 1);
    REQUIRE(state.count("compute", "expire_display") == 1);

    state.configure("expire_slow").computeDelay = std::chrono::milliseconds(120);
    state.configure("expire_slow").computeResults = {Result::YIELD};
    const auto epoch = Scheduler::Impl::BeginComputeRound();
    computeCycles(fixture.scheduler, 1);

    // Measure only the idle wait: the slow module's sleep may overshoot on CI.
    const auto start = std::chrono::steady_clock::now();
    Scheduler::Impl::WaitComputeRound(epoch, std::chrono::milliseconds(100));
    REQUIRE(std::chrono::steady_clock::now() - start < std::chrono::milliseconds(50));
    REQUIRE(state.count("compute", "expire_display") == 1);

    state.configure("expire_slow").computeDelay = std::chrono::milliseconds(0);
    computeCycles(fixture.scheduler, 1);
    REQUIRE(state.count("compute", "expire_display") == 2);

    REQUIRE(fixture.shutdown() == Result::SUCCESS);
}

TEST_CASE("Scheduler keeps join inputs frame aligned behind a view",
          "[core][runtime][scheduler][dataflow][alias]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("aligned_source").counter = true;
    state.configure("aligned_view").inputs = {"in"};
    state.configure("aligned_view").aliasInput = "in";
    state.configure("aligned_late").counter = true;
    state.configure("aligned_late").pendingComputeResults = {Result::TIMEOUT, Result::TIMEOUT};
    state.configure("aligned_join").inputs = {"left", "right"};
    state.configure("aligned_ticker");

    SchedulerFixture fixture;
    auto source = fixture.module("aligned_source");
    auto view = fixture.module("aligned_view", linkOutputs({{"in", source}}));
    auto late = fixture.module("aligned_late");
    auto join = fixture.module("aligned_join", linkOutputs({{"left", view}, {"right", late}}));
    auto ticker = fixture.module("aligned_ticker");
    for (const auto& module : {source, view, late, join, ticker}) {
        REQUIRE(fixture.scheduler.add(module) == Result::SUCCESS);
    }

    state.clearEvents();
    computeCycles(fixture.scheduler, 5);

    using Reads = std::vector<std::vector<F32>>;
    REQUIRE(state.reads["aligned_join"] == Reads{{1.0f, 1.0f}, {2.0f, 2.0f}, {3.0f, 3.0f}});

    REQUIRE(fixture.shutdown() == Result::SUCCESS);
}

TEST_CASE("Scheduler discards a view frame whose sibling from the same source skipped",
          "[core][runtime][scheduler][dataflow][alias]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("rejoin_source").counter = true;
    state.configure("rejoin_view").inputs = {"in"};
    state.configure("rejoin_view").aliasInput = "in";
    state.configure("rejoin_decimator").inputs = {"in"};
    state.configure("rejoin_decimator").forwardInput = "in";
    state.configure("rejoin_decimator").computeResults = {
        Result::SKIP, Result::SUCCESS, Result::SKIP, Result::SUCCESS,
    };
    state.configure("rejoin_join").inputs = {"left", "right"};

    SchedulerFixture fixture;
    auto source = fixture.module("rejoin_source");
    auto view = fixture.module("rejoin_view", linkOutputs({{"in", source}}));
    auto decimator = fixture.module("rejoin_decimator", linkOutputs({{"in", source}}));
    auto join = fixture.module("rejoin_join", linkOutputs({{"left", view}, {"right", decimator}}));
    for (const auto& module : {source, view, decimator, join}) {
        REQUIRE(fixture.scheduler.add(module) == Result::SUCCESS);
    }

    state.clearEvents();
    computeCycles(fixture.scheduler, 4);

    using Reads = std::vector<std::vector<F32>>;
    REQUIRE(state.count("compute", "rejoin_source") == 4);
    REQUIRE(state.reads["rejoin_join"] == Reads{{2.0f, 2.0f}, {4.0f, 4.0f}});

    REQUIRE(fixture.shutdown() == Result::SUCCESS);
}

TEST_CASE("Scheduler paces a fork through a view by its slowest consumer",
          "[core][runtime][scheduler][dataflow][alias]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("lag_source").counter = true;
    state.configure("lag_view").inputs = {"in"};
    state.configure("lag_view").aliasInput = "in";
    state.configure("lag_display").inputs = {"in"};
    state.configure("lag_display").computeResults = {Result::YIELD, Result::YIELD};
    state.configure("lag_fast").inputs = {"in"};
    state.configure("lag_ticker");

    SchedulerFixture fixture;
    auto source = fixture.module("lag_source");
    auto view = fixture.module("lag_view", linkOutputs({{"in", source}}));
    auto display = fixture.module("lag_display", linkOutputs({{"in", view}}));
    auto fast = fixture.module("lag_fast", linkOutputs({{"in", source}}));
    auto ticker = fixture.module("lag_ticker");
    for (const auto& module : {source, view, display, fast, ticker}) {
        REQUIRE(fixture.scheduler.add(module) == Result::SUCCESS);
    }

    state.clearEvents();
    computeCycles(fixture.scheduler, 3);

    using Reads = std::vector<std::vector<F32>>;
    REQUIRE(state.count("compute", "lag_source") == 1);
    REQUIRE(state.reads["lag_fast"] == Reads{{1.0f}});
    REQUIRE(state.reads["lag_display"] == Reads{{1.0f}, {1.0f}, {1.0f}});

    computeCycles(fixture.scheduler, 1);
    REQUIRE(state.count("compute", "lag_source") == 2);
    REQUIRE(state.reads["lag_fast"] == Reads{{1.0f}, {2.0f}});
    REQUIRE(state.reads["lag_display"] == Reads{{1.0f}, {1.0f}, {1.0f}, {2.0f}});

    REQUIRE(fixture.shutdown() == Result::SUCCESS);
}

TEST_CASE("Scheduler never shows a throttled view consumer the same frame twice",
          "[core][runtime][scheduler][dataflow][alias][throttle]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("frames_source").counter = true;
    state.configure("frames_view").inputs = {"in"};
    state.configure("frames_view").aliasInput = "in";
    state.configure("frames_display").inputs = {"in"};
    state.configure("frames_display").taint = Module::Taint::THROTTLED;

    SchedulerFixture fixture;
    auto source = fixture.module("frames_source", {}, RuntimeType::PYTHON);
    auto view = fixture.module("frames_view", linkOutputs({{"in", source}}));
    auto display = fixture.module("frames_display", linkOutputs({{"in", view}}));
    for (const auto& module : {source, view, display}) {
        REQUIRE(fixture.scheduler.add(module) == Result::SUCCESS);
    }

    state.clearEvents();
    computeCycles(fixture.scheduler, 2);

    // Keep the deadline crossing inside the source's compute submission.
    state.configure("frames_source").computeDelay = std::chrono::milliseconds(120);
    computeCycles(fixture.scheduler, 1);
    state.configure("frames_source").computeDelay = std::chrono::milliseconds(0);
    computeCycles(fixture.scheduler, 3);

    using Reads = std::vector<std::vector<F32>>;
    REQUIRE(state.reads["frames_display"] == Reads{{1.0f}, {4.0f}});

    REQUIRE(fixture.shutdown() == Result::SUCCESS);
}

TEST_CASE("Scheduler keeps a view frame when the sibling that read it is itself waiting",
          "[core][runtime][scheduler][dataflow][alias]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("waiting_source").counter = true;
    state.configure("waiting_view").inputs = {"in"};
    state.configure("waiting_view").aliasInput = "in";
    state.configure("waiting_copy").inputs = {"in"};
    state.configure("waiting_copy").forwardInput = "in";
    state.configure("waiting_late").counter = true;
    state.configure("waiting_late").pendingComputeResults = {Result::TIMEOUT, Result::TIMEOUT};
    state.configure("waiting_join").inputs = {"view", "copy", "late"};
    state.configure("waiting_ticker");

    SchedulerFixture fixture;
    auto source = fixture.module("waiting_source");
    auto view = fixture.module("waiting_view", linkOutputs({{"in", source}}));
    auto copy = fixture.module("waiting_copy", linkOutputs({{"in", source}}));
    auto late = fixture.module("waiting_late");
    auto join = fixture.module("waiting_join", linkOutputs({{"view", view}, {"copy", copy}, {"late", late}}));
    auto ticker = fixture.module("waiting_ticker");
    for (const auto& module : {source, view, copy, late, join, ticker}) {
        REQUIRE(fixture.scheduler.add(module) == Result::SUCCESS);
    }

    state.clearEvents();
    computeCycles(fixture.scheduler, 5);

    using Reads = std::vector<std::vector<F32>>;
    REQUIRE(state.reads["waiting_join"] == Reads{{1.0f, 1.0f, 1.0f}, {2.0f, 2.0f, 2.0f}, {3.0f, 3.0f, 3.0f}});

    REQUIRE(fixture.shutdown() == Result::SUCCESS);
}

TEST_CASE("Scheduler discards a copied frame whose sibling from the same source skipped",
          "[core][runtime][scheduler][dataflow][lineage]") {
    for (const auto runtime : {RuntimeType::NATIVE, RuntimeType::PYTHON}) {
        DYNAMIC_SECTION("Squelch runtime: " << runtime) {
            auto& state = syntheticState();
            state.reset();
            state.configure("squelch_source").counter = true;
            state.configure("squelch_copy").inputs = {"in"};
            state.configure("squelch_copy").forwardInput = "in";
            state.configure("squelch_gate").inputs = {"in"};
            state.configure("squelch_gate").forwardInput = "in";
            state.configure("squelch_gate").computeResults = {
                Result::SKIP, Result::SUCCESS, Result::SKIP, Result::SUCCESS, Result::SKIP, Result::SUCCESS,
            };
            state.configure("squelch_join").inputs = {"left", "right"};

            SchedulerFixture fixture;
            auto source = fixture.module("squelch_source");
            auto copy = fixture.module("squelch_copy", linkOutputs({{"in", source}}));
            auto gate = fixture.module("squelch_gate", linkOutputs({{"in", source}}), runtime);
            auto join = fixture.module("squelch_join", linkOutputs({{"left", copy}, {"right", gate}}));
            for (const auto& module : {source, copy, gate, join}) {
                REQUIRE(fixture.scheduler.add(module) == Result::SUCCESS);
            }

            state.clearEvents();
            computeCycles(fixture.scheduler, 6);

            using Reads = std::vector<std::vector<F32>>;
            REQUIRE(state.count("compute", "squelch_source") == 6);
            REQUIRE(state.reads["squelch_join"] == Reads{{2.0f, 2.0f}, {4.0f, 4.0f}, {6.0f, 6.0f}});

            REQUIRE(fixture.shutdown() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Scheduler discards related frames when a skip happens further upstream",
          "[core][runtime][scheduler][dataflow][lineage]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("deep_source").counter = true;
    state.configure("deep_gate").inputs = {"in"};
    state.configure("deep_gate").forwardInput = "in";
    state.configure("deep_gate").computeResults = {Result::SKIP, Result::SUCCESS, Result::SKIP, Result::SUCCESS};
    state.configure("deep_filter").inputs = {"in"};
    state.configure("deep_filter").forwardInput = "in";
    state.configure("deep_copy").inputs = {"in"};
    state.configure("deep_copy").forwardInput = "in";
    state.configure("deep_join").inputs = {"left", "right"};
    state.configure("deep_join").forwardInput = "left";
    state.configure("deep_sink").inputs = {"in"};
    state.configure("deep_sink").forwardInput = "in";
    state.configure("deep_other").inputs = {"in"};
    state.configure("deep_other").forwardInput = "in";
    state.configure("deep_final").inputs = {"left", "right"};

    SchedulerFixture fixture;
    auto source = fixture.module("deep_source");
    auto gate = fixture.module("deep_gate", linkOutputs({{"in", source}}));
    auto filter = fixture.module("deep_filter", linkOutputs({{"in", gate}}), RuntimeType::PYTHON);
    auto copy = fixture.module("deep_copy", linkOutputs({{"in", source}}));
    auto join = fixture.module("deep_join", linkOutputs({{"left", filter}, {"right", copy}}));
    auto sink = fixture.module("deep_sink", linkOutputs({{"in", join}}));
    auto other = fixture.module("deep_other", linkOutputs({{"in", source}}));
    auto final = fixture.module("deep_final", linkOutputs({{"left", sink}, {"right", other}}));
    for (const auto& module : {source, gate, filter, copy, join, sink, other, final}) {
        REQUIRE(fixture.scheduler.add(module) == Result::SUCCESS);
    }

    state.clearEvents();
    computeCycles(fixture.scheduler, 4);

    using Reads = std::vector<std::vector<F32>>;
    REQUIRE(state.count("compute", "deep_source") == 4);
    REQUIRE(state.reads["deep_join"] == Reads{{2.0f, 2.0f}, {4.0f, 4.0f}});
    REQUIRE(state.reads["deep_final"] == Reads{{2.0f, 2.0f}, {4.0f, 4.0f}});

    REQUIRE(fixture.shutdown() == Result::SUCCESS);
}

TEST_CASE("Scheduler drops a frame from an independent source when a sibling skips",
          "[core][runtime][scheduler][dataflow][lineage]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("independent_radio").counter = true;
    state.configure("independent_gate").inputs = {"in"};
    state.configure("independent_gate").forwardInput = "in";
    state.configure("independent_gate").computeResults = {Result::SKIP};
    state.configure("independent_other").counter = true;
    state.configure("independent_join").inputs = {"left", "right"};

    SchedulerFixture fixture;
    auto radio = fixture.module("independent_radio");
    auto gate = fixture.module("independent_gate", linkOutputs({{"in", radio}}));
    auto other = fixture.module("independent_other");
    auto join = fixture.module("independent_join", linkOutputs({{"left", gate}, {"right", other}}));
    for (const auto& module : {radio, gate, other, join}) {
        REQUIRE(fixture.scheduler.add(module) == Result::SUCCESS);
    }

    state.clearEvents();
    computeCycles(fixture.scheduler, 3);

    using Reads = std::vector<std::vector<F32>>;
    REQUIRE(state.reads["independent_join"] == Reads{{2.0f, 2.0f}, {3.0f, 3.0f}});

    REQUIRE(fixture.shutdown() == Result::SUCCESS);
}

TEST_CASE("Scheduler drops every held input of a join fed by mixed sources when one skips",
          "[core][runtime][scheduler][dataflow][lineage]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("mixed_radio").counter = true;
    state.configure("mixed_gate").inputs = {"in"};
    state.configure("mixed_gate").forwardInput = "in";
    state.configure("mixed_gate").computeResults = {Result::SKIP};
    state.configure("mixed_copy").inputs = {"in"};
    state.configure("mixed_copy").forwardInput = "in";
    state.configure("mixed_other").counter = true;
    state.configure("mixed_join").inputs = {"gate", "copy", "other"};

    SchedulerFixture fixture;
    auto radio = fixture.module("mixed_radio");
    auto gate = fixture.module("mixed_gate", linkOutputs({{"in", radio}}));
    auto copy = fixture.module("mixed_copy", linkOutputs({{"in", radio}}));
    auto other = fixture.module("mixed_other");
    auto join = fixture.module("mixed_join", linkOutputs({{"gate", gate}, {"copy", copy}, {"other", other}}));
    for (const auto& module : {radio, gate, copy, other, join}) {
        REQUIRE(fixture.scheduler.add(module) == Result::SUCCESS);
    }

    state.clearEvents();
    computeCycles(fixture.scheduler, 3);

    using Reads = std::vector<std::vector<F32>>;
    REQUIRE(state.reads["mixed_join"] == Reads{{2.0f, 2.0f, 2.0f}, {3.0f, 3.0f, 3.0f}});

    REQUIRE(fixture.shutdown() == Result::SUCCESS);
}

TEST_CASE("Scheduler discards related frames at a join that was still waiting on another source",
          "[core][runtime][scheduler][dataflow][lineage]") {
    for (const bool view : {true, false}) {
        DYNAMIC_SECTION("Related branch is a " << (view ? "view" : "copy")) {
            auto& state = syntheticState();
            state.reset();
            state.configure("waiting_join_source").counter = true;
            state.configure("waiting_join_branch").inputs = {"in"};
            if (view) {
                state.configure("waiting_join_branch").aliasInput = "in";
            } else {
                state.configure("waiting_join_branch").forwardInput = "in";
            }
            state.configure("waiting_join_gate").inputs = {"in"};
            state.configure("waiting_join_gate").forwardInput = "in";
            state.configure("waiting_join_gate").computeResults = {Result::SKIP};
            state.configure("waiting_join_late").counter = true;
            state.configure("waiting_join_late").pendingComputeResults = {Result::TIMEOUT, Result::TIMEOUT};
            state.configure("waiting_join").inputs = {"branch", "gate", "late"};
            state.configure("waiting_join_ticker");

            SchedulerFixture fixture;
            auto source = fixture.module("waiting_join_source");
            auto branch = fixture.module("waiting_join_branch", linkOutputs({{"in", source}}));
            auto gate = fixture.module("waiting_join_gate", linkOutputs({{"in", source}}));
            auto late = fixture.module("waiting_join_late");
            auto join = fixture.module("waiting_join",
                                       linkOutputs({{"branch", branch}, {"gate", gate}, {"late", late}}));
            auto ticker = fixture.module("waiting_join_ticker");
            for (const auto& module : {source, branch, gate, late, join, ticker}) {
                REQUIRE(fixture.scheduler.add(module) == Result::SUCCESS);
            }

            state.clearEvents();
            computeCycles(fixture.scheduler, 4);

            using Reads = std::vector<std::vector<F32>>;
            REQUIRE(state.reads["waiting_join"] == Reads{{2.0f, 2.0f, 2.0f}});

            REQUIRE(fixture.shutdown() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Scheduler never shows a throttled consumer a frame its view skipped",
          "[core][runtime][scheduler][dataflow][alias][throttle]") {
    for (const U64 views : {0, 1}) {
        DYNAMIC_SECTION("Views after the gate: " << views) {
            auto& state = syntheticState();
            state.reset();
            state.configure("skipped_view_source").counter = true;
            state.configure("skipped_view_gate").inputs = {"in"};
            state.configure("skipped_view_gate").aliasInput = "in";
            state.configure("skipped_view_gate").computeResults = {Result::SUCCESS, Result::SUCCESS, Result::SKIP};
            state.configure("skipped_view_display").inputs = {"in"};
            state.configure("skipped_view_display").taint = Module::Taint::THROTTLED;

            SchedulerFixture fixture;
            auto source = fixture.module("skipped_view_source");
            auto gate = fixture.module("skipped_view_gate", linkOutputs({{"in", source}}));
            REQUIRE(fixture.scheduler.add(source) == Result::SUCCESS);
            REQUIRE(fixture.scheduler.add(gate) == Result::SUCCESS);

            auto upstream = gate;
            for (U64 index = 0; index < views; ++index) {
                const std::string name = "skipped_view_reshape_" + std::to_string(index);
                state.configure(name).inputs = {"in"};
                state.configure(name).aliasInput = "in";
                auto reshape = fixture.module(name, linkOutputs({{"in", upstream}}));
                REQUIRE(fixture.scheduler.add(reshape) == Result::SUCCESS);
                upstream = reshape;
            }

            auto display = fixture.module("skipped_view_display", linkOutputs({{"in", upstream}}));
            REQUIRE(fixture.scheduler.add(display) == Result::SUCCESS);

            state.clearEvents();
            computeCycles(fixture.scheduler, 3);
            REQUIRE(state.count("compute", "skipped_view_source") == 3);

            std::this_thread::sleep_for(std::chrono::milliseconds(120));
            computeCycles(fixture.scheduler, 2);

            using Reads = std::vector<std::vector<F32>>;
            REQUIRE(state.reads["skipped_view_display"] == Reads{{1.0f}, {4.0f}});

            REQUIRE(fixture.shutdown() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Scheduler cascades integrations that skip until a window completes",
          "[core][runtime][scheduler][dataflow][integration]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("cascade_source").counter = true;
    state.configure("cascade_first").inputs = {"in"};
    state.configure("cascade_first").integrate = 4;
    state.configure("cascade_second").inputs = {"in"};
    state.configure("cascade_second").integrate = 4;
    state.configure("cascade_sink").inputs = {"in"};

    SchedulerFixture fixture;
    auto source = fixture.module("cascade_source");
    auto first = fixture.module("cascade_first", linkOutputs({{"in", source}}));
    auto second = fixture.module("cascade_second", linkOutputs({{"in", first}}));
    auto sink = fixture.module("cascade_sink", linkOutputs({{"in", second}}));
    for (const auto& module : {source, first, second, sink}) {
        REQUIRE(fixture.scheduler.add(module) == Result::SUCCESS);
    }

    state.clearEvents();
    computeCycles(fixture.scheduler, 32);

    using Reads = std::vector<std::vector<F32>>;
    REQUIRE(state.count("compute", "cascade_source") == 32);
    REQUIRE(state.count("compute", "cascade_first") == 32);
    REQUIRE(state.count("compute", "cascade_second") == 8);
    REQUIRE(state.reads["cascade_sink"] == Reads{{136.0f}, {392.0f}});

    REQUIRE(fixture.shutdown() == Result::SUCCESS);
}

TEST_CASE("Scheduler never shows a throttled consumer a partial integration",
          "[core][runtime][scheduler][dataflow][integration][throttle]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("partial_source").counter = true;
    state.configure("partial_integrator").inputs = {"in"};
    state.configure("partial_integrator").integrate = 4;
    state.configure("partial_display").inputs = {"in"};
    state.configure("partial_display").taint = Module::Taint::THROTTLED;

    SchedulerFixture fixture;
    auto source = fixture.module("partial_source");
    auto integrator = fixture.module("partial_integrator", linkOutputs({{"in", source}}));
    auto display = fixture.module("partial_display", linkOutputs({{"in", integrator}}));
    for (const auto& module : {source, integrator, display}) {
        REQUIRE(fixture.scheduler.add(module) == Result::SUCCESS);
    }

    state.clearEvents();
    for (U64 round = 0; round < 3; ++round) {
        computeCycles(fixture.scheduler, 6);
        std::this_thread::sleep_for(std::chrono::milliseconds(120));
    }
    computeCycles(fixture.scheduler, 6);

    const auto& reads = state.reads["partial_display"];
    REQUIRE(reads.size() >= 3);
    for (const auto& read : reads) {
        CAPTURE(read.front());
        REQUIRE(static_cast<U64>(read.front() + 6.0f) % 16 == 0);
    }

    REQUIRE(fixture.shutdown() == Result::SUCCESS);
}

TEST_CASE("Scheduler pairs integrations of different lengths on the frame they share",
          "[core][runtime][scheduler][dataflow][integration][lineage]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("rates_source").counter = true;
    state.configure("rates_short").inputs = {"in"};
    state.configure("rates_short").integrate = 2;
    state.configure("rates_long").inputs = {"in"};
    state.configure("rates_long").integrate = 4;
    state.configure("rates_join").inputs = {"short", "long"};

    SchedulerFixture fixture;
    auto source = fixture.module("rates_source");
    auto shortIntegrator = fixture.module("rates_short", linkOutputs({{"in", source}}));
    auto longIntegrator = fixture.module("rates_long", linkOutputs({{"in", source}}));
    auto join = fixture.module("rates_join", linkOutputs({{"short", shortIntegrator}, {"long", longIntegrator}}));
    for (const auto& module : {source, shortIntegrator, longIntegrator, join}) {
        REQUIRE(fixture.scheduler.add(module) == Result::SUCCESS);
    }

    state.clearEvents();
    computeCycles(fixture.scheduler, 8);

    using Reads = std::vector<std::vector<F32>>;
    REQUIRE(state.count("compute", "rates_source") == 8);
    REQUIRE(state.reads["rates_join"] == Reads{{7.0f, 10.0f}, {15.0f, 26.0f}});

    REQUIRE(fixture.shutdown() == Result::SUCCESS);
}

TEST_CASE("Scheduler realigns a join behind a throttled path that skips frames",
          "[core][runtime][scheduler][dataflow][lineage][throttle]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("realign_source").counter = true;
    state.configure("realign_slow").inputs = {"in"};
    state.configure("realign_slow").forwardInput = "in";
    state.configure("realign_slow").taint = Module::Taint::THROTTLED;
    state.configure("realign_copy").inputs = {"in"};
    state.configure("realign_copy").forwardInput = "in";
    state.configure("realign_join").inputs = {"slow", "copy"};
    state.configure("realign_ticker");

    SchedulerFixture fixture;
    auto source = fixture.module("realign_source");
    auto slow = fixture.module("realign_slow", linkOutputs({{"in", source}}));
    auto copy = fixture.module("realign_copy", linkOutputs({{"in", source}}));
    auto join = fixture.module("realign_join", linkOutputs({{"slow", slow}, {"copy", copy}}));
    auto ticker = fixture.module("realign_ticker");
    for (const auto& module : {source, slow, copy, join, ticker}) {
        REQUIRE(fixture.scheduler.add(module) == Result::SUCCESS);
    }

    state.clearEvents();
    for (U64 round = 0; round < 3; ++round) {
        computeCycles(fixture.scheduler, 4);
        std::this_thread::sleep_for(std::chrono::milliseconds(120));
    }
    computeCycles(fixture.scheduler, 4);

    const auto& reads = state.reads["realign_join"];
    REQUIRE(reads.size() >= 3);
    for (const auto& read : reads) {
        REQUIRE(read[0] == read[1]);
    }
    REQUIRE(reads.back()[0] > reads.front()[0]);

    REQUIRE(fixture.shutdown() == Result::SUCCESS);
}

TEST_CASE("Scheduler never shows a throttled consumer a view that stalled after its producer moved on",
          "[core][runtime][scheduler][dataflow][alias][throttle]") {
    for (const auto status : {Result::YIELD, Result::TIMEOUT}) {
        DYNAMIC_SECTION("Stall result: " << status) {
            auto& state = syntheticState();
            state.reset();
            state.configure("stalled_view_source").counter = true;
            state.configure("stalled_view").inputs = {"in"};
            state.configure("stalled_view").aliasInput = "in";
            state.configure("stalled_view").computeResults = {Result::SUCCESS, Result::SUCCESS, status};
            state.configure("stalled_view_display").inputs = {"in"};
            state.configure("stalled_view_display").taint = Module::Taint::THROTTLED;

            SchedulerFixture fixture;
            auto source = fixture.module("stalled_view_source");
            auto view = fixture.module("stalled_view", linkOutputs({{"in", source}}));
            auto display = fixture.module("stalled_view_display", linkOutputs({{"in", view}}));
            for (const auto& module : {source, view, display}) {
                REQUIRE(fixture.scheduler.add(module) == Result::SUCCESS);
            }

            state.clearEvents();
            computeCycles(fixture.scheduler, 3);
            std::this_thread::sleep_for(std::chrono::milliseconds(120));
            computeCycles(fixture.scheduler, 2);
            std::this_thread::sleep_for(std::chrono::milliseconds(120));
            computeCycles(fixture.scheduler, 1);

            using Reads = std::vector<std::vector<F32>>;
            REQUIRE(state.reads["stalled_view_display"] == Reads{{1.0f}, {3.0f}, {5.0f}});

            REQUIRE(fixture.shutdown() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Scheduler waits for a related view that has not produced yet before discarding a skipped frame",
          "[core][runtime][scheduler][dataflow][lineage][alias]") {
    for (const auto status : {Result::YIELD, Result::TIMEOUT}) {
        DYNAMIC_SECTION("View stall result: " << status) {
            auto& state = syntheticState();
            state.reset();
            state.configure("unborn_source").counter = true;
            state.configure("unborn_gate").inputs = {"in"};
            state.configure("unborn_gate").forwardInput = "in";
            state.configure("unborn_gate").computeResults = {Result::SKIP};
            state.configure("unborn_view").inputs = {"in"};
            state.configure("unborn_view").aliasInput = "in";
            state.configure("unborn_view").computeResults = {status};
            state.configure("unborn_join").inputs = {"gate", "view"};
            state.configure("unborn_ticker");

            SchedulerFixture fixture;
            auto source = fixture.module("unborn_source");
            auto gate = fixture.module("unborn_gate", linkOutputs({{"in", source}}));
            auto view = fixture.module("unborn_view", linkOutputs({{"in", source}}));
            auto join = fixture.module("unborn_join", linkOutputs({{"gate", gate}, {"view", view}}));
            auto ticker = fixture.module("unborn_ticker");
            for (const auto& module : {source, gate, view, join, ticker}) {
                REQUIRE(fixture.scheduler.add(module) == Result::SUCCESS);
            }

            state.clearEvents();
            computeCycles(fixture.scheduler, 4);

            using Reads = std::vector<std::vector<F32>>;
            REQUIRE(state.count("compute", "unborn_source") == 3);
            REQUIRE(state.reads["unborn_join"] == Reads{{2.0f, 2.0f}, {3.0f, 3.0f}});

            REQUIRE(fixture.shutdown() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Scheduler discards every input tied to a skipped frame through shared sources",
          "[core][runtime][scheduler][dataflow][lineage]") {
    using Slots = std::vector<std::string>;
    for (const auto& slots : {Slots{"gate", "mix", "view"}, Slots{"a_gate", "b_view", "c_mix"}}) {
        DYNAMIC_SECTION("Join slots: " << slots[0] << ", " << slots[1] << ", " << slots[2]) {
            const auto& gateSlot = slots[0];
            const auto& mixSlot = slots[0] == "gate" ? slots[1] : slots[2];
            const auto& viewSlot = slots[0] == "gate" ? slots[2] : slots[1];

            auto& state = syntheticState();
            state.reset();
            state.configure("tied_x").counter = true;
            state.configure("tied_y").counter = true;
            state.configure("tied_gate").inputs = {"in"};
            state.configure("tied_gate").forwardInput = "in";
            state.configure("tied_gate").computeResults = {Result::SKIP};
            state.configure("tied_mix").inputs = {"x", "y"};
            state.configure("tied_mix").forwardInput = "y";
            state.configure("tied_view").inputs = {"in"};
            state.configure("tied_view").aliasInput = "in";
            state.configure("tied_join").inputs = {gateSlot, mixSlot, viewSlot};

            SchedulerFixture fixture;
            auto x = fixture.module("tied_x");
            auto y = fixture.module("tied_y");
            auto gate = fixture.module("tied_gate", linkOutputs({{"in", x}}));
            auto mix = fixture.module("tied_mix", linkOutputs({{"x", x}, {"y", y}}));
            auto view = fixture.module("tied_view", linkOutputs({{"in", y}}));
            auto join = fixture.module("tied_join", linkOutputs({{gateSlot, gate}, {mixSlot, mix}, {viewSlot, view}}));
            for (const auto& module : {x, y, gate, mix, view, join}) {
                REQUIRE(fixture.scheduler.add(module) == Result::SUCCESS);
            }

            state.clearEvents();
            computeCycles(fixture.scheduler, 3);

            using Reads = std::vector<std::vector<F32>>;
            REQUIRE(state.count("compute", "tied_x") == 3);
            REQUIRE(state.count("compute", "tied_y") == 3);
            REQUIRE(state.reads["tied_join"] == Reads{{2.0f, 2.0f, 2.0f}, {3.0f, 3.0f, 3.0f}});

            REQUIRE(fixture.shutdown() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Scheduler pairs independent streams that share a constant source without discarding",
          "[core][runtime][scheduler][dataflow][lineage][static]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("shared_taps").taint = Module::Taint::STATIC_OUTPUT;
    state.configure("shared_radio_a").counter = true;
    state.configure("shared_filter_a").inputs = {"in", "taps"};
    state.configure("shared_filter_a").forwardInput = "in";
    state.configure("shared_slow_a").inputs = {"in"};
    state.configure("shared_slow_a").forwardInput = "in";
    state.configure("shared_slow_a").computeResults = {Result::YIELD, Result::YIELD};
    state.configure("shared_radio_b").counter = true;
    state.configure("shared_filter_b").inputs = {"in", "taps"};
    state.configure("shared_filter_b").forwardInput = "in";
    state.configure("shared_join").inputs = {"a", "b"};
    state.configure("shared_ticker");

    SchedulerFixture fixture;
    auto taps = fixture.module("shared_taps");
    auto radioA = fixture.module("shared_radio_a");
    auto filterA = fixture.module("shared_filter_a", linkOutputs({{"in", radioA}, {"taps", taps}}));
    auto slowA = fixture.module("shared_slow_a", linkOutputs({{"in", filterA}}));
    auto radioB = fixture.module("shared_radio_b");
    auto filterB = fixture.module("shared_filter_b", linkOutputs({{"in", radioB}, {"taps", taps}}));
    auto join = fixture.module("shared_join", linkOutputs({{"a", slowA}, {"b", filterB}}));
    auto ticker = fixture.module("shared_ticker");
    for (const auto& module : {taps, radioA, filterA, slowA, radioB, filterB, join, ticker}) {
        REQUIRE(fixture.scheduler.add(module) == Result::SUCCESS);
    }

    state.clearEvents();
    computeCycles(fixture.scheduler, 5);

    using Reads = std::vector<std::vector<F32>>;
    REQUIRE(state.count("compute", "shared_taps") == 1);
    REQUIRE(state.reads["shared_join"] == Reads{{1.0f, 1.0f}, {2.0f, 2.0f}, {3.0f, 3.0f}});

    REQUIRE(fixture.shutdown() == Result::SUCCESS);
}

TEST_CASE("Scheduler forgets a cancelled selection after stop and start",
          "[core][runtime][scheduler][dataflow]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("cancelled_source");
    state.configure("cancelled_sink").inputs = {"in"};

    SchedulerFixture fixture;
    auto source = fixture.module("cancelled_source");
    auto sink = fixture.module("cancelled_sink", linkOutputs({{"in", source}}));
    REQUIRE(fixture.scheduler.add(source) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.add(sink) == Result::SUCCESS);

    bool stopped = false;
    state.configure("cancelled_source").pendingCompute = [&] {
        if (stopped) {
            return Result::TIMEOUT;
        }
        stopped = true;
        REQUIRE(fixture.scheduler.stop() == Result::SUCCESS);
        return Result::SUCCESS;
    };

    state.clearEvents();
    std::unordered_set<std::string> failedModules;
    REQUIRE(fixture.scheduler.compute(failedModules) == Result::SUCCESS);
    REQUIRE(state.count("compute", "cancelled_source") == 0);

    REQUIRE(fixture.scheduler.start() == Result::SUCCESS);
    REQUIRE(fixture.scheduler.compute(failedModules) == Result::SUCCESS);
    REQUIRE(failedModules.empty());
    REQUIRE(state.count("compute", "cancelled_source") == 0);
    REQUIRE(state.count("compute", "cancelled_sink") == 0);

    REQUIRE(fixture.shutdown() == Result::SUCCESS);
}

TEST_CASE("Scheduler keeps aligning a join that mixes two sources with a throttled path",
          "[core][runtime][scheduler][dataflow][lineage][throttle]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("torture_a").counter = true;
    state.configure("torture_b").counter = true;
    state.configure("torture_gate").inputs = {"in"};
    state.configure("torture_gate").forwardInput = "in";
    state.configure("torture_gate").computeResults = {
        Result::SUCCESS, Result::SUCCESS, Result::SKIP, Result::SUCCESS, Result::SUCCESS, Result::SKIP,
    };
    state.configure("torture_copy_a").inputs = {"in"};
    state.configure("torture_copy_a").forwardInput = "in";
    state.configure("torture_mix").inputs = {"a", "b"};
    state.configure("torture_mix").forwardInput = "b";
    state.configure("torture_copy_b").inputs = {"in"};
    state.configure("torture_copy_b").forwardInput = "in";
    state.configure("torture_slow_b").inputs = {"in"};
    state.configure("torture_slow_b").forwardInput = "in";
    state.configure("torture_slow_b").taint = Module::Taint::THROTTLED;
    state.configure("torture_checker").inputs = {"gate", "copy_a", "mix", "copy_b", "slow_b"};

    SchedulerFixture fixture;
    auto a = fixture.module("torture_a");
    auto b = fixture.module("torture_b");
    auto gate = fixture.module("torture_gate", linkOutputs({{"in", a}}));
    auto copyA = fixture.module("torture_copy_a", linkOutputs({{"in", a}}));
    auto mix = fixture.module("torture_mix", linkOutputs({{"a", a}, {"b", b}}));
    auto copyB = fixture.module("torture_copy_b", linkOutputs({{"in", b}}));
    auto slowB = fixture.module("torture_slow_b", linkOutputs({{"in", b}}));
    auto checker = fixture.module("torture_checker", linkOutputs({
        {"gate", gate}, {"copy_a", copyA}, {"mix", mix}, {"copy_b", copyB}, {"slow_b", slowB},
    }));
    for (const auto& module : {a, b, gate, copyA, mix, copyB, slowB, checker}) {
        REQUIRE(fixture.scheduler.add(module) == Result::SUCCESS);
    }

    state.clearEvents();
    for (U64 round = 0; round < 4; ++round) {
        computeCycles(fixture.scheduler, 8);
        std::this_thread::sleep_for(std::chrono::milliseconds(120));
    }
    computeCycles(fixture.scheduler, 8);

    const auto& reads = state.reads["torture_checker"];
    REQUIRE(reads.size() >= 3);

    F32 last = 0.0f;
    for (const auto& read : reads) {
        CAPTURE(read[0], read[1], read[2], read[3], read[4]);
        REQUIRE(read[0] == read[1]);
        REQUIRE(read[2] == read[3]);
        REQUIRE(read[3] == read[4]);
        REQUIRE(read[1] > last);
        last = read[1];
    }

    REQUIRE(fixture.shutdown() == Result::SUCCESS);
}

TEST_CASE("Scheduler realigns nested joins after a partial skip upstream",
          "[core][runtime][scheduler][dataflow][lineage][alias]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("nested_x").counter = true;
    state.configure("nested_x").computeResults = {Result::SKIP};
    state.configure("nested_y").counter = true;
    state.configure("nested_mix").inputs = {"x", "y"};
    state.configure("nested_mix").aliasInput = "y";
    state.configure("nested_view").inputs = {"in"};
    state.configure("nested_view").aliasInput = "in";
    state.configure("nested_join").inputs = {"mix", "view"};
    state.configure("nested_ticker");

    SchedulerFixture fixture;
    auto x = fixture.module("nested_x");
    auto y = fixture.module("nested_y");
    auto mix = fixture.module("nested_mix", linkOutputs({{"x", x}, {"y", y}}));
    auto view = fixture.module("nested_view", linkOutputs({{"in", y}}));
    auto join = fixture.module("nested_join", linkOutputs({{"mix", mix}, {"view", view}}));
    auto ticker = fixture.module("nested_ticker");
    for (const auto& module : {x, y, mix, view, join, ticker}) {
        REQUIRE(fixture.scheduler.add(module) == Result::SUCCESS);
    }

    state.clearEvents();
    computeCycles(fixture.scheduler, 8);

    const auto& reads = state.reads["nested_join"];
    REQUIRE(reads.size() >= 3);
    for (const auto& read : reads) {
        REQUIRE(read[0] == read[1]);
    }
    REQUIRE(state.count("compute", "nested_y") >= 4);

    REQUIRE(fixture.shutdown() == Result::SUCCESS);
}

TEST_CASE("Scheduler recovers two joins that share sources after repeated skips",
          "[core][runtime][scheduler][dataflow][lineage]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("cross_a").counter = true;
    state.configure("cross_b").counter = true;
    state.configure("cross_first").inputs = {"a", "b"};
    state.configure("cross_gate").inputs = {"in"};
    state.configure("cross_gate").forwardInput = "in";
    state.configure("cross_gate").computeResults = {Result::SKIP, Result::SKIP};
    state.configure("cross_second").inputs = {"a", "gate"};

    SchedulerFixture fixture;
    auto a = fixture.module("cross_a");
    auto b = fixture.module("cross_b");
    auto first = fixture.module("cross_first", linkOutputs({{"a", a}, {"b", b}}));
    auto gate = fixture.module("cross_gate", linkOutputs({{"in", b}}));
    auto second = fixture.module("cross_second", linkOutputs({{"a", a}, {"gate", gate}}));
    for (const auto& module : {a, b, first, gate, second}) {
        REQUIRE(fixture.scheduler.add(module) == Result::SUCCESS);
    }

    state.clearEvents();
    computeCycles(fixture.scheduler, 6);

    REQUIRE(state.count("compute", "cross_a") >= 5);
    REQUIRE(state.count("compute", "cross_b") >= 5);
    REQUIRE(state.reads["cross_first"].size() >= 5);
    REQUIRE(state.reads["cross_second"].size() >= 3);

    REQUIRE(fixture.shutdown() == Result::SUCCESS);
}

TEST_CASE("Scheduler recovers two joins when the independent input runs in the join's segment",
          "[core][runtime][scheduler][dataflow][lineage]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("segment_cross_a").counter = true;
    state.configure("segment_cross_b").counter = true;
    state.configure("segment_cross_first").inputs = {"a", "b"};
    state.configure("segment_cross_gate").inputs = {"in"};
    state.configure("segment_cross_gate").forwardInput = "in";
    state.configure("segment_cross_gate").computeResults = {Result::SKIP, Result::SKIP};
    state.configure("segment_cross_copy").inputs = {"in"};
    state.configure("segment_cross_copy").forwardInput = "in";
    state.configure("segment_cross_second").inputs = {"copy", "gate"};

    SchedulerFixture fixture;
    auto a = fixture.module("segment_cross_a");
    auto b = fixture.module("segment_cross_b");
    auto first = fixture.module("segment_cross_first", linkOutputs({{"a", a}, {"b", b}}));
    auto gate = fixture.module("segment_cross_gate", linkOutputs({{"in", b}}));
    auto copy = fixture.module("segment_cross_copy", linkOutputs({{"in", a}}), RuntimeType::PYTHON);
    auto second = fixture.module("segment_cross_second",
                                 linkOutputs({{"copy", copy}, {"gate", gate}}),
                                 RuntimeType::PYTHON);
    for (const auto& module : {a, b, first, gate, copy, second}) {
        REQUIRE(fixture.scheduler.add(module) == Result::SUCCESS);
    }

    state.clearEvents();
    computeCycles(fixture.scheduler, 8);

    REQUIRE(state.count("compute", "segment_cross_a") >= 6);
    REQUIRE(state.count("compute", "segment_cross_b") >= 6);
    REQUIRE(state.reads["segment_cross_second"].size() >= 3);

    REQUIRE(fixture.shutdown() == Result::SUCCESS);
}

TEST_CASE("Scheduler tells downstream joins about frames discarded during alignment",
          "[core][runtime][scheduler][dataflow][lineage][throttle]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("discard_a").counter = true;
    state.configure("discard_b").counter = true;
    state.configure("discard_b").taint = Module::Taint::THROTTLED;
    state.configure("discard_gate").inputs = {"in"};
    state.configure("discard_gate").forwardInput = "in";
    state.configure("discard_gate").computeResults = {Result::SUCCESS, Result::SKIP};
    state.configure("discard_mix").inputs = {"a", "b"};
    state.configure("discard_mix").forwardInput = "b";
    state.configure("discard_join").inputs = {"gate", "mix"};
    state.configure("discard_join").forwardInput = "mix";
    state.configure("discard_rejoin").inputs = {"b", "join"};

    SchedulerFixture fixture;
    auto a = fixture.module("discard_a");
    auto b = fixture.module("discard_b");
    auto gate = fixture.module("discard_gate", linkOutputs({{"in", a}}));
    auto mix = fixture.module("discard_mix", linkOutputs({{"a", a}, {"b", b}}));
    auto join = fixture.module("discard_join", linkOutputs({{"gate", gate}, {"mix", mix}}));
    auto rejoin = fixture.module("discard_rejoin", linkOutputs({{"b", b}, {"join", join}}));
    for (const auto& module : {a, b, gate, mix, join, rejoin}) {
        REQUIRE(fixture.scheduler.add(module) == Result::SUCCESS);
    }

    state.clearEvents();
    for (U64 round = 0; round < 5; ++round) {
        computeCycles(fixture.scheduler, 6);
        std::this_thread::sleep_for(std::chrono::milliseconds(120));
    }
    computeCycles(fixture.scheduler, 6);

    REQUIRE(state.count("compute", "discard_b") >= 5);
    REQUIRE(state.count("compute", "discard_rejoin") >= 4);
    for (const auto& read : state.reads["discard_rejoin"]) {
        REQUIRE(read[0] == read[1]);
    }

    REQUIRE(fixture.shutdown() == Result::SUCCESS);
}

TEST_CASE("Scheduler keeps a split behind a squelch that stays closed flowing",
          "[core][runtime][scheduler][dataflow]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("closed_source").counter = true;
    state.configure("closed_copy").inputs = {"in"};
    state.configure("closed_copy").forwardInput = "in";
    state.configure("closed_gate").inputs = {"in"};
    state.configure("closed_gate").forwardInput = "in";
    state.configure("closed_gate").skipping = true;
    state.configure("closed_join").inputs = {"copy", "gate"};

    SchedulerFixture fixture;
    auto source = fixture.module("closed_source");
    auto copy = fixture.module("closed_copy", linkOutputs({{"in", source}}));
    auto gate = fixture.module("closed_gate", linkOutputs({{"in", source}}), RuntimeType::PYTHON);
    auto join = fixture.module("closed_join", linkOutputs({{"copy", copy}, {"gate", gate}}));
    for (const auto& module : {source, copy, gate, join}) {
        REQUIRE(fixture.scheduler.add(module) == Result::SUCCESS);
    }

    computeCycles(fixture.scheduler, 2000);
    REQUIRE(state.count("compute", "closed_join") == 0);
    REQUIRE(state.count("compute", "closed_source") > 1000);

    REQUIRE(fixture.shutdown() == Result::SUCCESS);
}

TEST_CASE("Scheduler keeps frame identity below constant sources",
          "[core][runtime][scheduler][dataflow][static]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("identity_constant").taint = Module::Taint::STATIC_OUTPUT;
    state.configure("identity_counter").inputs = {"in"};
    state.configure("identity_counter").counter = true;
    state.configure("identity_copy").inputs = {"in"};
    state.configure("identity_copy").forwardInput = "in";
    state.configure("identity_copy").computeResults = {Result::YIELD};
    state.configure("identity_gate").inputs = {"in"};
    state.configure("identity_gate").forwardInput = "in";
    state.configure("identity_gate").computeResults = {Result::SKIP};
    state.configure("identity_join").inputs = {"copy", "gate"};
    state.configure("identity_ticker");

    SchedulerFixture fixture;
    auto constant = fixture.module("identity_constant");
    auto counter = fixture.module("identity_counter", linkOutputs({{"in", constant}}));
    auto copy = fixture.module("identity_copy", linkOutputs({{"in", counter}}));
    auto gate = fixture.module("identity_gate", linkOutputs({{"in", counter}}));
    auto join = fixture.module("identity_join", linkOutputs({{"copy", copy}, {"gate", gate}}));
    auto ticker = fixture.module("identity_ticker");
    for (const auto& module : {constant, counter, copy, gate, join, ticker}) {
        REQUIRE(fixture.scheduler.add(module) == Result::SUCCESS);
    }

    state.clearEvents();
    computeCycles(fixture.scheduler, 8);

    const auto& reads = state.reads["identity_join"];
    REQUIRE(reads.size() >= 3);
    for (const auto& read : reads) {
        CAPTURE(read[0], read[1]);
        REQUIRE(read[0] == read[1]);
    }

    REQUIRE(fixture.shutdown() == Result::SUCCESS);
}

TEST_CASE("Scheduler keeps a join flowing behind a closed gate and a stalling worker",
          "[core][runtime][scheduler][dataflow]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("stalling_source").counter = true;
    state.configure("stalling_gate").inputs = {"in"};
    state.configure("stalling_gate").forwardInput = "in";
    state.configure("stalling_gate").skipping = true;
    state.configure("stalling_worker").inputs = {"in"};
    state.configure("stalling_worker").forwardInput = "in";
    state.configure("stalling_worker").alternating = true;
    state.configure("stalling_join").inputs = {"gate", "worker"};

    SchedulerFixture fixture;
    auto source = fixture.module("stalling_source");
    auto gate = fixture.module("stalling_gate", linkOutputs({{"in", source}}));
    auto worker = fixture.module("stalling_worker", linkOutputs({{"in", source}}));
    auto join = fixture.module("stalling_join", linkOutputs({{"gate", gate}, {"worker", worker}}));
    for (const auto& module : {source, gate, worker, join}) {
        REQUIRE(fixture.scheduler.add(module) == Result::SUCCESS);
    }

    computeCycles(fixture.scheduler, 2000);
    REQUIRE(state.count("compute", "stalling_source") > 900);

    REQUIRE(fixture.shutdown() == Result::SUCCESS);
}

TEST_CASE("Scheduler keeps independent pairings identical across joins after a late start",
          "[core][runtime][scheduler][dataflow][lineage]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("late_a").counter = true;
    state.configure("late_a").pendingComputeResults = {Result::TIMEOUT};
    state.configure("late_b").counter = true;
    state.configure("late_gate").inputs = {"in"};
    state.configure("late_gate").forwardInput = "in";
    state.configure("late_gate").computeResults = {Result::SKIP};
    state.configure("late_first").inputs = {"a", "b"};
    state.configure("late_first").forwardInput = "b";
    state.configure("late_second").inputs = {"a", "b"};
    state.configure("late_second").forwardInput = "b";
    state.configure("late_final").inputs = {"first", "second"};

    SchedulerFixture fixture;
    auto a = fixture.module("late_a");
    auto b = fixture.module("late_b");
    auto gate = fixture.module("late_gate", linkOutputs({{"in", b}}));
    auto first = fixture.module("late_first", linkOutputs({{"a", a}, {"b", b}}));
    auto second = fixture.module("late_second", linkOutputs({{"a", a}, {"b", gate}}));
    auto final = fixture.module("late_final", linkOutputs({{"first", first}, {"second", second}}));
    for (const auto& module : {a, b, gate, first, second, final}) {
        REQUIRE(fixture.scheduler.add(module) == Result::SUCCESS);
    }

    state.clearEvents();
    computeCycles(fixture.scheduler, 200);

    REQUIRE(state.count("compute", "late_a") >= 190);
    REQUIRE(state.count("compute", "late_b") >= 190);
    REQUIRE(state.count("compute", "late_final") >= 190);
    for (const auto& read : state.reads["late_final"]) {
        REQUIRE(read[0] == read[1]);
    }

    REQUIRE(fixture.shutdown() == Result::SUCCESS);
}

TEST_CASE("Scheduler does not let an idle flowgraph slow another one down",
          "[core][runtime][scheduler][dataflow][wake]") {
    skipSchedulerTimingOnCI();
    auto& state = syntheticState();
    state.reset();
    state.configure("idle_constant").taint = Module::Taint::STATIC_OUTPUT;
    state.configure("idle_stateless").inputs = {"in"};
    state.configure("idle_stateless").taint = Module::Taint::STATELESS;
    state.configure("busy_source");
    state.configure("busy_sink").inputs = {"in"};

    SchedulerFixture idle;
    auto constant = idle.module("idle_constant");
    auto stateless = idle.module("idle_stateless", linkOutputs({{"in", constant}}));
    REQUIRE(idle.scheduler.add(constant) == Result::SUCCESS);
    REQUIRE(idle.scheduler.add(stateless) == Result::SUCCESS);

    SchedulerFixture busy;
    auto source = busy.module("busy_source");
    auto sink = busy.module("busy_sink", linkOutputs({{"in", source}}));
    REQUIRE(busy.scheduler.add(source) == Result::SUCCESS);
    REQUIRE(busy.scheduler.add(sink) == Result::SUCCESS);

    state.clearEvents();
    std::unordered_set<std::string> failedModules;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(300);
    while (std::chrono::steady_clock::now() < deadline) {
        const auto epoch = Scheduler::Impl::BeginComputeRound();
        REQUIRE(idle.scheduler.compute(failedModules) == Result::SUCCESS);
        REQUIRE(busy.scheduler.compute(failedModules) == Result::SUCCESS);
        Scheduler::Impl::WaitComputeRound(epoch, std::chrono::milliseconds(100));
    }

    REQUIRE(state.count("compute", "idle_constant") == 1);
    REQUIRE(state.count("compute", "busy_sink") > 1000);

    REQUIRE(busy.shutdown() == Result::SUCCESS);
    REQUIRE(idle.shutdown() == Result::SUCCESS);
}

TEST_CASE("Scheduler keeps a live join running next to a settled module with a live input",
          "[core][runtime][scheduler][dataflow][static]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("settled_live");
    state.configure("settled_constant").inputs = {"in"};
    state.configure("settled_constant").taint = Module::Taint::STATIC_OUTPUT;
    state.configure("settled_join").inputs = {"live", "constant"};

    SchedulerFixture fixture;
    auto live = fixture.module("settled_live");
    auto constant = fixture.module("settled_constant", linkOutputs({{"in", live}}));
    auto join = fixture.module("settled_join", linkOutputs({{"live", live}, {"constant", constant}}));
    for (const auto& module : {live, constant, join}) {
        REQUIRE(fixture.scheduler.add(module) == Result::SUCCESS);
    }

    state.clearEvents();
    computeCycles(fixture.scheduler, 10);

    REQUIRE(state.count("compute", "settled_constant") == 1);
    REQUIRE(state.count("compute", "settled_live") == 10);
    REQUIRE(state.count("compute", "settled_join") == 10);

    REQUIRE(fixture.shutdown() == Result::SUCCESS);
}

TEST_CASE("Scheduler keeps compute rounds separate per compute thread",
          "[core][runtime][scheduler][wake]") {
    skipSchedulerTimingOnCI();
    const auto epoch = Scheduler::Impl::BeginComputeRound();
    Scheduler::Impl::ReportComputeProgress();

    std::thread other([] {
        const auto otherEpoch = Scheduler::Impl::BeginComputeRound();
        Scheduler::Impl::WaitComputeRound(otherEpoch, std::chrono::milliseconds(1));
    });
    other.join();

    const auto start = std::chrono::steady_clock::now();
    Scheduler::Impl::WaitComputeRound(epoch, std::chrono::milliseconds(100));
    REQUIRE(std::chrono::steady_clock::now() - start < std::chrono::milliseconds(50));
}

TEST_CASE("Scheduler recomputes static outputs that were waiting unread when settlement is cleared",
          "[core][runtime][scheduler][dataflow][static]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("stale_constant").counter = true;
    state.configure("stale_constant").taint = Module::Taint::STATIC_OUTPUT;
    state.configure("stale_transform").inputs = {"in"};
    state.configure("stale_transform").forwardInput = "in";
    state.configure("stale_transform").taint = Module::Taint::STATELESS;
    state.configure("stale_live").pendingComputeResults = {Result::TIMEOUT, Result::TIMEOUT, Result::TIMEOUT};
    state.configure("stale_join").inputs = {"constant", "live"};
    state.configure("stale_ticker");

    SchedulerFixture fixture;
    auto constant = fixture.module("stale_constant");
    auto transform = fixture.module("stale_transform", linkOutputs({{"in", constant}}));
    auto live = fixture.module("stale_live");
    auto join = fixture.module("stale_join", linkOutputs({{"constant", transform}, {"live", live}}));
    auto ticker = fixture.module("stale_ticker");
    for (const auto& module : {constant, transform, live, join, ticker}) {
        REQUIRE(fixture.scheduler.add(module) == Result::SUCCESS);
    }

    state.clearEvents();
    computeCycles(fixture.scheduler, 2);
    REQUIRE(state.count("compute", "stale_constant") == 1);
    REQUIRE(state.count("compute", "stale_join") == 0);

    REQUIRE(fixture.scheduler.synchronize([] { return Result::SUCCESS; }) == Result::SUCCESS);
    computeCycles(fixture.scheduler, 4);

    REQUIRE(state.count("compute", "stale_constant") == 2);
    REQUIRE_FALSE(state.reads["stale_join"].empty());
    for (const auto& read : state.reads["stale_join"]) {
        REQUIRE(read[0] == 2.0f);
    }

    REQUIRE(fixture.shutdown() == Result::SUCCESS);
}

TEST_CASE("Scheduler start wakes an idle compute round",
          "[core][runtime][scheduler][wake]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("restart_source");

    SchedulerFixture fixture;
    auto source = fixture.module("restart_source");
    REQUIRE(fixture.scheduler.add(source) == Result::SUCCESS);
    REQUIRE(fixture.scheduler.stop() == Result::SUCCESS);

    std::promise<void> idle;
    auto idleReady = idle.get_future();
    std::unordered_set<std::string> failedModules;
    auto waiter = std::async(std::launch::async, [&] {
        const auto epoch = Scheduler::Impl::BeginComputeRound();
        const auto result = fixture.scheduler.compute(failedModules);
        idle.set_value();
        Scheduler::Impl::WaitComputeRound(epoch, std::chrono::seconds(30));
        return result;
    });

    // Restart only after the stopped scheduler has completed its idle round.
    // Keep the fallback wait longer than the wake-up deadline so a missed
    // notification still fails without depending on millisecond-scale timing.
    idleReady.wait();
    const bool wasWaiting = waiter.wait_for(std::chrono::milliseconds(0)) ==
                            std::future_status::timeout;
    const auto startResult = fixture.scheduler.start();
    const bool woke = waiter.wait_for(std::chrono::seconds(5)) ==
                      std::future_status::ready;
    if (!woke) {
        Scheduler::Context::NotifyPendingCompute();
    }
    const auto computeResult = waiter.get();

    REQUIRE(wasWaiting);
    REQUIRE(startResult == Result::SUCCESS);
    REQUIRE(woke);
    REQUIRE(computeResult == Result::SUCCESS);
    REQUIRE(failedModules.empty());
    REQUIRE(state.count("compute", "restart_source") == 0);

    computeCycles(fixture.scheduler, 1);
    REQUIRE(state.count("compute", "restart_source") == 1);

    REQUIRE(fixture.shutdown() == Result::SUCCESS);
}

TEST_CASE("Scheduler recomputes a static output whose view was waiting unread when settlement is cleared",
          "[core][runtime][scheduler][dataflow][static][alias]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("stale_view_constant").counter = true;
    state.configure("stale_view_constant").taint = Module::Taint::STATIC_OUTPUT;
    state.configure("stale_view_squelch").inputs = {"in"};
    state.configure("stale_view_squelch").aliasInput = "in";
    state.configure("stale_view_live").pendingComputeResults = {Result::TIMEOUT, Result::TIMEOUT, Result::TIMEOUT};
    state.configure("stale_view_join").inputs = {"constant", "squelch", "live"};
    state.configure("stale_view_ticker");

    SchedulerFixture fixture;
    auto constant = fixture.module("stale_view_constant");
    auto squelch = fixture.module("stale_view_squelch", linkOutputs({{"in", constant}}));
    auto live = fixture.module("stale_view_live");
    auto join = fixture.module("stale_view_join",
                               linkOutputs({{"constant", constant}, {"squelch", squelch}, {"live", live}}));
    auto ticker = fixture.module("stale_view_ticker");
    for (const auto& module : {constant, squelch, live, join, ticker}) {
        REQUIRE(fixture.scheduler.add(module) == Result::SUCCESS);
    }

    state.clearEvents();
    computeCycles(fixture.scheduler, 2);
    REQUIRE(state.count("compute", "stale_view_join") == 0);

    REQUIRE(fixture.scheduler.synchronize([] { return Result::SUCCESS; }) == Result::SUCCESS);
    computeCycles(fixture.scheduler, 20);

    REQUIRE(state.count("compute", "stale_view_constant") == 2);
    REQUIRE(state.count("compute", "stale_view_join") >= 10);
    for (const auto& read : state.reads["stale_view_join"]) {
        REQUIRE(read[0] == 2.0f);
        REQUIRE(read[1] == 2.0f);
    }

    REQUIRE(fixture.shutdown() == Result::SUCCESS);
}

TEST_CASE("Scheduler keeps copied and aliased branches paired after settlement is cleared",
          "[core][runtime][scheduler][dataflow][static][alias]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("paired_constant").counter = true;
    state.configure("paired_constant").taint = Module::Taint::STATIC_OUTPUT;
    state.configure("paired_view").inputs = {"in"};
    state.configure("paired_view").aliasInput = "in";
    state.configure("paired_copy").inputs = {"in"};
    state.configure("paired_copy").forwardInput = "in";
    state.configure("paired_live").counter = true;
    state.configure("paired_live").pendingComputeResults = {Result::TIMEOUT, Result::TIMEOUT};
    state.configure("paired_left").inputs = {"view", "live"};
    state.configure("paired_left").forwardInput = "live";
    state.configure("paired_right").inputs = {"copy", "live"};
    state.configure("paired_right").forwardInput = "live";
    state.configure("paired_compare").inputs = {"left", "right"};
    state.configure("paired_ticker");

    SchedulerFixture fixture;
    auto constant = fixture.module("paired_constant");
    auto view = fixture.module("paired_view", linkOutputs({{"in", constant}}));
    auto copy = fixture.module("paired_copy", linkOutputs({{"in", view}}));
    auto live = fixture.module("paired_live");
    auto left = fixture.module("paired_left", linkOutputs({{"view", view}, {"live", live}}));
    auto right = fixture.module("paired_right", linkOutputs({{"copy", copy}, {"live", live}}));
    auto compare = fixture.module("paired_compare", linkOutputs({{"left", left}, {"right", right}}));
    auto ticker = fixture.module("paired_ticker");
    for (const auto& module : {constant, view, copy, live, left, right, compare, ticker}) {
        REQUIRE(fixture.scheduler.add(module) == Result::SUCCESS);
    }

    state.clearEvents();
    computeCycles(fixture.scheduler, 1);

    REQUIRE(fixture.scheduler.synchronize([] { return Result::SUCCESS; }) == Result::SUCCESS);
    computeCycles(fixture.scheduler, 50);

    REQUIRE(state.count("compute", "paired_compare") >= 40);
    for (const auto& read : state.reads["paired_compare"]) {
        REQUIRE(read[0] == read[1]);
    }

    REQUIRE(fixture.shutdown() == Result::SUCCESS);
}

TEST_CASE("Scheduler keeps the throttle cadence of a module that stalls",
          "[core][runtime][scheduler][dataflow][throttle]") {
    for (const auto status : {Result::YIELD, Result::TIMEOUT}) {
        DYNAMIC_SECTION("Stall result: " << status) {
            auto& state = syntheticState();
            state.reset();
            state.configure("cadence_throttled").taint = Module::Taint::THROTTLED;
            state.configure("cadence_throttled").computeResults = std::vector<Result>(2000, status);
            state.configure("cadence_live");

            SchedulerFixture fixture;
            auto throttled = fixture.module("cadence_throttled");
            auto live = fixture.module("cadence_live");
            REQUIRE(fixture.scheduler.add(throttled) == Result::SUCCESS);
            REQUIRE(fixture.scheduler.add(live) == Result::SUCCESS);

            state.clearEvents();
            const auto start = std::chrono::steady_clock::now();
            computeCycles(fixture.scheduler, 1000);
            const auto elapsed = std::chrono::steady_clock::now() - start;

            REQUIRE(state.count("compute", "cadence_live") == 1000);
            if (elapsed < std::chrono::milliseconds(100)) {
                REQUIRE(state.count("compute", "cadence_throttled") == 1);
            }

            REQUIRE(fixture.shutdown() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Scheduler retries the held input of a throttled module that stalled",
          "[core][runtime][scheduler][dataflow][throttle]") {
    for (const auto runtime : {RuntimeType::NATIVE, RuntimeType::PYTHON}) {
        for (const auto status : {Result::YIELD, Result::TIMEOUT}) {
            DYNAMIC_SECTION("Runtime: " << runtime << ", stall result: " << status) {
                auto& state = syntheticState();
                state.reset();
                bool emitted = false;
                state.configure("retry_source").counter = true;
                state.configure("retry_source").pendingCompute = [&emitted] {
                    if (emitted) {
                        return Result::TIMEOUT;
                    }
                    emitted = true;
                    return Result::SUCCESS;
                };
                state.configure("retry_consumer").inputs = {"in"};
                state.configure("retry_consumer").forwardInput = "in";
                state.configure("retry_consumer").taint = Module::Taint::THROTTLED;
                state.configure("retry_consumer").computeResults = {status};
                state.configure("retry_sink").inputs = {"in"};
                state.configure("retry_ticker");

                SchedulerFixture fixture;
                auto source = fixture.module("retry_source");
                auto consumer = fixture.module("retry_consumer", linkOutputs({{"in", source}}), runtime);
                auto sink = fixture.module("retry_sink", linkOutputs({{"in", consumer}}));
                auto ticker = fixture.module("retry_ticker");
                for (const auto& module : {source, consumer, sink, ticker}) {
                    REQUIRE(fixture.scheduler.add(module) == Result::SUCCESS);
                }

                state.clearEvents();
                computeCycles(fixture.scheduler, 2);
                std::this_thread::sleep_for(std::chrono::milliseconds(120));
                computeCycles(fixture.scheduler, 20);

                using Reads = std::vector<std::vector<F32>>;
                REQUIRE(state.count("compute", "retry_consumer") == 2);
                REQUIRE(state.reads["retry_sink"] == Reads{{1.0f}});

                REQUIRE(fixture.shutdown() == Result::SUCCESS);
            }
        }
    }
}

TEST_CASE("Scheduler starts throttle deadlines when a module actually runs",
          "[core][runtime][scheduler][dataflow][throttle]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("late_deadline_source");
    state.configure("late_deadline_slow").inputs = {"in"};
    state.configure("late_deadline_slow").computeDelay = std::chrono::milliseconds(150);
    state.configure("late_deadline_display").inputs = {"in"};
    state.configure("late_deadline_display").taint = Module::Taint::THROTTLED;

    SchedulerFixture fixture;
    auto source = fixture.module("late_deadline_source");
    auto slow = fixture.module("late_deadline_slow", linkOutputs({{"in", source}}));
    auto display = fixture.module("late_deadline_display", linkOutputs({{"in", slow}}), RuntimeType::PYTHON);
    for (const auto& module : {source, slow, display}) {
        REQUIRE(fixture.scheduler.add(module) == Result::SUCCESS);
    }

    state.clearEvents();
    computeCycles(fixture.scheduler, 1);
    REQUIRE(state.count("compute", "late_deadline_display") == 1);

    state.configure("late_deadline_slow").computeDelay = std::chrono::milliseconds(0);
    computeCycles(fixture.scheduler, 3);
    REQUIRE(state.count("compute", "late_deadline_display") == 1);

    REQUIRE(fixture.shutdown() == Result::SUCCESS);
}

TEST_CASE("Scheduler keeps a held copy's metadata on the frame it was computed from",
          "[core][runtime][scheduler][dataflow][alias]") {
    auto& state = syntheticState();
    state.reset();
    state.configure("metadata_source").counter = true;
    state.configure("metadata_copy").inputs = {"in"};
    state.configure("metadata_copy").forwardInput = "in";
    state.configure("metadata_copy").inheritInput = "in";
    state.configure("metadata_consumer").inputs = {"in"};
    state.configure("metadata_consumer").computeResults = {Result::YIELD};
    state.configure("metadata_consumer").readsFrame = true;
    state.configure("metadata_ticker");

    SchedulerFixture fixture;
    auto source = fixture.module("metadata_source");
    auto copy = fixture.module("metadata_copy", linkOutputs({{"in", source}}));
    auto consumer = fixture.module("metadata_consumer", linkOutputs({{"in", copy}}));
    auto ticker = fixture.module("metadata_ticker");
    for (const auto& module : {source, copy, consumer, ticker}) {
        REQUIRE(fixture.scheduler.add(module) == Result::SUCCESS);
    }

    REQUIRE(copy->outputs().at("out").tensor.inheritsAttributes(copy->inputs().at("in").tensor));

    state.clearEvents();
    computeCycles(fixture.scheduler, 4);

    const auto& reads = state.reads["metadata_consumer"];
    REQUIRE(reads.size() >= 3);
    for (const auto& read : reads) {
        REQUIRE(read.size() == 2);
        CAPTURE(read[0], read[1]);
        REQUIRE(read[0] == read[1]);
    }

    REQUIRE(fixture.shutdown() == Result::SUCCESS);
}

TEST_CASE("Scheduler keeps derived metadata of a held frame on the frame it was computed from",
          "[core][runtime][scheduler][dataflow][alias]") {
    for (const bool cloned : {false, true}) {
        DYNAMIC_SECTION("Published " << (cloned ? "as a clone" : "directly")) {
            auto& state = syntheticState();
            state.reset();
            state.configure("derived_constant").taint = Module::Taint::STATIC_OUTPUT;
            state.configure("derived_source").counter = true;
            state.configure("derived_mix").inputs = {"a", "b"};
            state.configure("derived_mix").forwardInput = "b";
            state.configure("derived_mix").inheritInput = "a";
            state.configure("derived_mix").deriveFrame = "b";
            state.configure("derived_mix").publishClone = cloned;
            state.configure("derived_consumer").inputs = {"in"};
            state.configure("derived_consumer").computeResults = {Result::YIELD};
            state.configure("derived_consumer").readsFrame = true;
            state.configure("derived_ticker");

            SchedulerFixture fixture;
            auto constant = fixture.module("derived_constant");
            auto source = fixture.module("derived_source");
            auto mix = fixture.module("derived_mix", linkOutputs({{"a", constant}, {"b", source}}));
            auto consumer = fixture.module("derived_consumer", linkOutputs({{"in", mix}}));
            auto ticker = fixture.module("derived_ticker");
            for (const auto& module : {constant, source, mix, consumer, ticker}) {
                REQUIRE(fixture.scheduler.add(module) == Result::SUCCESS);
            }

            state.clearEvents();
            computeCycles(fixture.scheduler, 4);

            const auto& reads = state.reads["derived_consumer"];
            REQUIRE(reads.size() >= 3);
            for (const auto& read : reads) {
                REQUIRE(read.size() == 2);
                CAPTURE(read[0], read[1]);
                REQUIRE(read[0] == read[1]);
            }

            REQUIRE(fixture.shutdown() == Result::SUCCESS);
        }
    }
}
