#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <any>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "jetstream/block_context.hh"
#include "jetstream/block_interface.hh"
#include "jetstream/detail/block_impl.hh"
#include "jetstream/detail/module_impl.hh"
#include "jetstream/flowgraph_view.hh"
#include "jetstream/memory/tensor.hh"
#include "jetstream/module_context.hh"
#include "jetstream/module_interface.hh"
#include "jetstream/registry.hh"
#include "jetstream/runtime_context_native_cpu.hh"
#include "jetstream/scheduler_context.hh"

namespace {

using namespace Jetstream;

constexpr auto kInitialValue = "initial";
constexpr auto kLifecycleProvider = "cyberether-lifecycle-worker";
constexpr auto kChildModuleType = "cyberether_lifecycle_child_module";
constexpr auto kMissingChildModuleType = "cyberether_lifecycle_missing_child_module";

enum class ConfigRole {
    Staged,
    Candidate,
};

enum class DuplicateInterface {
    None,
    Input,
    Output,
    Config,
    Metric,
};

Result DeserializeValue(const Parser::Map& data, std::string& value) {
    if (!data.contains("value")) {
        return Result::SUCCESS;
    }

    if (data.at("value").type() != typeid(std::string)) {
        return Result::ERROR;
    }

    value = std::any_cast<std::string>(data.at("value"));
    return Result::SUCCESS;
}

struct BlockProbe {
    Result candidateDeserializeResult = Result::SUCCESS;
    Result stagedDeserializeResult = Result::SUCCESS;
    Result validateResult = Result::SUCCESS;
    Result configureResult = Result::SUCCESS;
    Result defineResult = Result::SUCCESS;
    Result createResult = Result::SUCCESS;
    Result destroyResult = Result::SUCCESS;

    DuplicateInterface duplicate = DuplicateInterface::None;
    bool declareInput = false;
    bool declareOutput = false;
    bool declareConfig = false;
    bool declareMetric = false;
    bool produceOutput = false;
    bool produceChildOutput = false;
    bool exposeChildOutput = false;
    bool recordChildReconfigureEvents = false;
    bool recordSchedulerEvents = false;
    std::vector<std::string> children;
    std::string childModuleType = kChildModuleType;
    std::string failingChildCreate;
    std::string incompleteChildCreate;
    std::string failingChildDestroy;
    std::string failingChildValidate;
    std::string failingChildReconfigure;
    std::string failingChildInitialize;
    std::string failingChildPresentInitialize;
    std::string stagedDeserializeFailureValue;

    std::vector<std::string> events;
    std::vector<Block::State> hookStates;
    std::weak_ptr<Block> block;
    std::shared_ptr<Flowgraph::Environment> seenEnvironment;
    std::shared_ptr<Flowgraph::View> seenView;
    std::shared_ptr<Render::Window> seenRender;
};

struct SyntheticBlockConfig : Block::Config {
    std::shared_ptr<BlockProbe> probe;
    ConfigRole role = ConfigRole::Staged;
    std::string value = kInitialValue;
    Result serializeResult = Result::SUCCESS;

    std::string type() const override {
        return "cyberether_lifecycle_block";
    }

    std::string domain() const override {
        return "test";
    }

    std::string title() const override {
        return "Lifecycle Block";
    }

    std::string summary() const override {
        return "Synthetic lifecycle fixture.";
    }

    std::string description() const override {
        return "Exercises generic block orchestration.";
    }

    Result serialize(Parser::Map& data) const override {
        probe->events.push_back(role == ConfigRole::Staged
                                    ? "block.staged.serialize"
                                    : "block.candidate.serialize");
        if (serializeResult != Result::SUCCESS) {
            return serializeResult;
        }
        data["value"] = value;
        return Result::SUCCESS;
    }

    Result deserialize(const Parser::Map& data) override {
        const bool candidate = role == ConfigRole::Candidate;
        probe->events.push_back(candidate
                                    ? "block.candidate.deserialize"
                                    : "block.staged.deserialize");

        const auto result = candidate
                                ? probe->candidateDeserializeResult
                                : probe->stagedDeserializeResult;
        if (result != Result::SUCCESS) {
            return result;
        }

        if (!candidate &&
            data.contains("value") &&
            data.at("value").type() == typeid(std::string) &&
            std::any_cast<const std::string&>(data.at("value")) ==
                probe->stagedDeserializeFailureValue) {
            return Result::ERROR;
        }

        return DeserializeValue(data, value);
    }

    std::size_t hash() const override {
        return std::hash<std::string>{}(value);
    }
};

struct LifecycleChildConfig : Module::Config {
    std::string moduleType = kChildModuleType;
    std::string value = kInitialValue;

    std::string type() const override {
        return moduleType;
    }

    Result serialize(Parser::Map& data) const override {
        data["value"] = value;
        return Result::SUCCESS;
    }

    Result deserialize(const Parser::Map& data) override {
        return DeserializeValue(data, value);
    }

    std::size_t hash() const override {
        return std::hash<std::string>{}(value);
    }
};

struct LifecycleChildModule;

struct SyntheticBlockImpl : Block::Impl {
    std::shared_ptr<BlockProbe> probe;
    std::shared_ptr<SyntheticBlockConfig> staged;
    std::shared_ptr<SyntheticBlockConfig> candidate;
    std::unordered_map<std::string, std::shared_ptr<LifecycleChildConfig>> childConfigs;

    void recordHook(const std::string& event) {
        probe->events.push_back(event);
        if (const auto block = probe->block.lock()) {
            probe->hookStates.push_back(block->state());
        }
    }

    Result validate() override {
        recordHook("block.validate:" + candidate->value + ":" + staged->value);
        return probe->validateResult;
    }

    Result configure() override {
        recordHook("block.configure:" + staged->value);
        if (probe->configureResult != Result::SUCCESS) {
            return probe->configureResult;
        }

        for (auto& [_, config] : childConfigs) {
            config->value = staged->value;
        }
        return Result::SUCCESS;
    }

    Result define() override {
        recordHook("block.define:" + staged->value);
        if (probe->defineResult != Result::SUCCESS) {
            return probe->defineResult;
        }

        switch (probe->duplicate) {
            case DuplicateInterface::Input:
                JST_CHECK(defineInterfaceInput("port", "Port", "Synthetic input."));
                return defineInterfaceInput("port", "Port", "Synthetic input.");
            case DuplicateInterface::Output:
                JST_CHECK(defineInterfaceOutput("port", "Port", "Synthetic output."));
                return defineInterfaceOutput("port", "Port", "Synthetic output.");
            case DuplicateInterface::Config:
                JST_CHECK(defineInterfaceConfig("value", "Value", "Synthetic config.", "text"));
                return defineInterfaceConfig("value", "Value", "Synthetic config.", "text");
            case DuplicateInterface::Metric:
                JST_CHECK(defineInterfaceMetric("status",
                                                "Status",
                                                "Synthetic metric.",
                                                "text",
                                                [] { return std::any(std::string("ready")); }));
                return defineInterfaceMetric("status",
                                             "Status",
                                             "Synthetic metric.",
                                             "text",
                                             [] { return std::any(std::string("ready")); });
            case DuplicateInterface::None:
                break;
        }

        if (probe->declareOutput) {
            JST_CHECK(defineInterfaceOutput("out", "Output", "Synthetic output."));
        }

        if (probe->exposeChildOutput && !probe->declareOutput) {
            JST_CHECK(defineInterfaceOutput("out", "Output", "Synthetic child output."));
        }

        if (probe->declareInput) {
            JST_CHECK(defineInterfaceInput("in", "Input", "Synthetic input."));
        }

        if (probe->declareConfig) {
            JST_CHECK(defineInterfaceConfig("value", "Value", "Synthetic config.", "text"));
        }

        if (probe->declareMetric) {
            JST_CHECK(defineInterfaceMetric("status",
                                            "Status",
                                            "Synthetic metric.",
                                            "text",
                                            [] { return std::any(std::string("ready")); }));
        }

        return Result::SUCCESS;
    }

    Result create() override {
        recordHook("block.create:" + staged->value);

        childConfigs.clear();

        for (const auto& child : probe->children) {
            auto config = std::make_shared<LifecycleChildConfig>();
            config->moduleType = probe->childModuleType;
            config->value = staged->value;

            const auto result = moduleCreate(child, config, {});
            if (result == Result::SUCCESS || result == Result::INCOMPLETE) {
                childConfigs[child] = std::move(config);
            }
            if (result != Result::SUCCESS) {
                return result;
            }
        }

        if (probe->createResult != Result::SUCCESS) {
            return probe->createResult;
        }

        if (probe->produceOutput) {
            (void)outputs()["out"];
        }

        if (probe->exposeChildOutput) {
            if (probe->children.empty()) {
                return Result::ERROR;
            }
            JST_CHECK(moduleExposeOutput("out", {probe->children.front(), "out"}));
        }

        return Result::SUCCESS;
    }

    Result destroy() override {
        recordHook("block.destroy");
        return probe->destroyResult;
    }

    std::string childStagedValue(const std::string& child);
    std::string childCandidateValue(const std::string& child);
    std::string childConfigValue(const std::string& child) const;
    TensorLink childOutput(const std::string& child);
};

struct BlockBundle {
    std::shared_ptr<BlockProbe> probe;
    std::shared_ptr<SyntheticBlockConfig> staged;
    std::shared_ptr<SyntheticBlockConfig> candidate;
    std::shared_ptr<SyntheticBlockImpl> impl;
    std::shared_ptr<Block> block;
};

BlockBundle MakeBlock() {
    BlockBundle bundle;
    bundle.probe = std::make_shared<BlockProbe>();

    bundle.staged = std::make_shared<SyntheticBlockConfig>();
    bundle.staged->probe = bundle.probe;
    bundle.staged->role = ConfigRole::Staged;

    bundle.candidate = std::make_shared<SyntheticBlockConfig>();
    bundle.candidate->probe = bundle.probe;
    bundle.candidate->role = ConfigRole::Candidate;

    bundle.impl = std::make_shared<SyntheticBlockImpl>();
    bundle.impl->probe = bundle.probe;
    bundle.impl->staged = bundle.staged;
    bundle.impl->candidate = bundle.candidate;

    bundle.block = std::make_shared<Block>(bundle.impl, bundle.staged, bundle.candidate);
    bundle.probe->block = bundle.block;
    return bundle;
}

std::shared_ptr<Block::Context> MakeBlockContext(
    const std::shared_ptr<Scheduler>& scheduler = nullptr,
    const std::shared_ptr<Flowgraph::Environment>& environment = nullptr,
    const std::shared_ptr<Flowgraph::View>& view = nullptr,
    const std::shared_ptr<Render::Window>& render = nullptr) {
    return std::make_shared<Block::Context>(nullptr, render, scheduler, environment, view);
}

struct ModuleProbe {
    Result candidateDeserializeResult = Result::SUCCESS;
    Result stagedDeserializeResult = Result::SUCCESS;
    Result validateResult = Result::SUCCESS;
    Result defineResult = Result::SUCCESS;
    Result createResult = Result::SUCCESS;
    Result destroyResult = Result::SUCCESS;
    Result reconfigureResult = Result::SUCCESS;

    DuplicateInterface duplicate = DuplicateInterface::None;
    bool declareInput = false;
    bool declareOutput = false;
    bool produceOutput = false;
    bool commitReconfigure = true;
    bool commitBeforeReconfigureFailure = false;
    Module::Taint taint = Module::Taint::CLEAN;
    DeviceType expectedDevice = DeviceType::CPU;

    std::vector<std::string> events;
    std::vector<bool> identityReady;
};

struct SyntheticModuleConfig : Module::Config {
    std::shared_ptr<ModuleProbe> probe;
    ConfigRole role = ConfigRole::Staged;
    std::string value = kInitialValue;
    Result serializeResult = Result::SUCCESS;

    std::string type() const override {
        return "cyberether_lifecycle_module";
    }

    Result serialize(Parser::Map& data) const override {
        probe->events.push_back(role == ConfigRole::Staged
                                    ? "module.staged.serialize"
                                    : "module.candidate.serialize");
        if (serializeResult != Result::SUCCESS) {
            return serializeResult;
        }
        data["value"] = value;
        return Result::SUCCESS;
    }

    Result deserialize(const Parser::Map& data) override {
        const bool candidate = role == ConfigRole::Candidate;
        probe->events.push_back(candidate
                                    ? "module.candidate.deserialize"
                                    : "module.staged.deserialize");

        const auto result = candidate
                                ? probe->candidateDeserializeResult
                                : probe->stagedDeserializeResult;
        if (result != Result::SUCCESS) {
            return result;
        }

        return DeserializeValue(data, value);
    }

    std::size_t hash() const override {
        return std::hash<std::string>{}(value);
    }
};

struct SyntheticModuleImpl : Module::Impl,
                             NativeCpuRuntimeContext,
                             Scheduler::Context {
    std::shared_ptr<ModuleProbe> probe;
    std::shared_ptr<SyntheticModuleConfig> staged;
    std::shared_ptr<SyntheticModuleConfig> candidate;

    void recordHook(const std::string& event) {
        probe->events.push_back(event);
        probe->identityReady.push_back(name() == "lifecycle-module" &&
                                       device() == probe->expectedDevice &&
                                       runtime() == RuntimeType::NATIVE &&
                                       provider() == kLifecycleProvider);
    }

    Result validate() override {
        recordHook("module.validate:" + candidate->value + ":" + staged->value);
        return probe->validateResult;
    }

    Result define() override {
        recordHook("module.define:" + staged->value);
        if (probe->defineResult != Result::SUCCESS) {
            return probe->defineResult;
        }

        if (probe->taint != Module::Taint::CLEAN) {
            JST_CHECK(defineTaint(probe->taint));
        }

        if (probe->duplicate == DuplicateInterface::Input) {
            JST_CHECK(defineInterfaceInput("port"));
            return defineInterfaceInput("port");
        }

        if (probe->duplicate == DuplicateInterface::Output) {
            JST_CHECK(defineInterfaceOutput("port"));
            return defineInterfaceOutput("port");
        }

        if (probe->declareInput) {
            JST_CHECK(defineInterfaceInput("in"));
        }

        if (probe->declareOutput) {
            JST_CHECK(defineInterfaceOutput("out"));
        }

        return Result::SUCCESS;
    }

    Result create() override {
        recordHook("module.create:" + staged->value);
        if (probe->createResult != Result::SUCCESS) {
            return probe->createResult;
        }

        if (probe->produceOutput) {
            JST_CHECK(output.create(DeviceType::CPU, DataType::F32, {2}));
            output.at<F32>(0) = 3.0f;
            output.at<F32>(1) = 5.0f;
            outputs()["out"].produced(name(), "out", output);
        }

        return Result::SUCCESS;
    }

    Result destroy() override {
        recordHook("module.destroy:" + staged->value);
        return probe->destroyResult;
    }

    Result reconfigure() override {
        recordHook("module.reconfigure:" + candidate->value + ":" + staged->value);
        if (probe->commitReconfigure &&
            (probe->reconfigureResult == Result::SUCCESS ||
             probe->commitBeforeReconfigureFailure)) {
            staged->value = candidate->value;
        }
        return probe->reconfigureResult;
    }

    Tensor output;
};

struct ModuleBundle {
    std::shared_ptr<ModuleProbe> probe;
    std::shared_ptr<SyntheticModuleConfig> staged;
    std::shared_ptr<SyntheticModuleConfig> candidate;
    std::shared_ptr<SyntheticModuleImpl> impl;
    std::shared_ptr<Module> module;
};

ModuleBundle MakeModule(
    DeviceType device = DeviceType::CPU,
    const std::shared_ptr<Flowgraph::Environment>& environment = nullptr,
    const std::shared_ptr<Flowgraph::View>& view = nullptr) {
    ModuleBundle bundle;
    bundle.probe = std::make_shared<ModuleProbe>();
    bundle.probe->expectedDevice = device;

    bundle.staged = std::make_shared<SyntheticModuleConfig>();
    bundle.staged->probe = bundle.probe;
    bundle.staged->role = ConfigRole::Staged;

    bundle.candidate = std::make_shared<SyntheticModuleConfig>();
    bundle.candidate->probe = bundle.probe;
    bundle.candidate->role = ConfigRole::Candidate;

    bundle.impl = std::make_shared<SyntheticModuleImpl>();
    bundle.impl->probe = bundle.probe;
    bundle.impl->staged = bundle.staged;
    bundle.impl->candidate = bundle.candidate;

    const auto runtimeContext = std::static_pointer_cast<Runtime::Context>(bundle.impl);
    const auto schedulerContext = std::static_pointer_cast<Scheduler::Context>(bundle.impl);
    const auto context = std::make_shared<Module::Context>(runtimeContext,
                                                           schedulerContext,
                                                           environment,
                                                           view);

    bundle.module = std::make_shared<Module>(device,
                                              RuntimeType::NATIVE,
                                             kLifecycleProvider,
                                             bundle.impl,
                                             context,
                                             bundle.staged,
                                             bundle.candidate);
    return bundle;
}

struct LifecycleChildModule : Module::Impl,
                               NativeCpuRuntimeContext,
                               Scheduler::Context {
    std::shared_ptr<BlockProbe> probe;
    std::shared_ptr<LifecycleChildConfig> staged;
    std::shared_ptr<LifecycleChildConfig> candidate;

    bool matches(const std::string& child) const {
        return !child.empty() && name() == "lifecycle-block-" + child;
    }

    Result validate() override {
        if (probe->recordChildReconfigureEvents) {
            probe->events.push_back("child.validate:" + name() + ":" +
                                    candidate->value + ":" + staged->value);
        }
        return matches(probe->failingChildValidate) ? Result::ERROR : Result::SUCCESS;
    }

    Result define() override {
        if (probe->produceChildOutput) {
            JST_CHECK(defineInterfaceOutput("out"));
        }
        return Result::SUCCESS;
    }

    Result create() override {
        probe->events.push_back("child.create:" + name());
        probe->seenEnvironment = environment();
        probe->seenView = view();
        probe->seenRender = render();

        if (matches(probe->failingChildCreate)) {
            return Result::ERROR;
        }
        if (matches(probe->incompleteChildCreate)) {
            return Result::INCOMPLETE;
        }
        if (probe->produceChildOutput) {
            JST_CHECK(output.create(DeviceType::CPU, DataType::F32, {2}));
            output.at<F32>(0) = 7.0f;
            output.at<F32>(1) = 11.0f;
            outputs()["out"].produced(name(), "out", output);
        }
        return Result::SUCCESS;
    }

    Result destroy() override {
        probe->events.push_back("child.destroy:" + name());
        if (matches(probe->failingChildDestroy)) {
            return Result::ERROR;
        }
        return Result::SUCCESS;
    }

    Result reconfigure() override {
        if (probe->recordChildReconfigureEvents) {
            probe->events.push_back("child.reconfigure:" + name() + ":" +
                                    candidate->value + ":" + staged->value);
        }
        if (matches(probe->failingChildReconfigure)) {
            return Result::ERROR;
        }
        staged->value = candidate->value;
        return Result::SUCCESS;
    }

    Result presentInitialize() override {
        if (probe->recordSchedulerEvents) {
            probe->events.push_back("child.present_initialize:" + name());
        }
        return matches(probe->failingChildPresentInitialize)
                   ? Result::ERROR
                   : Result::SUCCESS;
    }

    Result computeInitialize() override {
        if (probe->recordSchedulerEvents) {
            probe->events.push_back("child.initialize:" + name());
        }
        return matches(probe->failingChildInitialize) ? Result::ERROR : Result::SUCCESS;
    }

    Result computeDeinitialize() override {
        if (probe->recordSchedulerEvents) {
            probe->events.push_back("child.deinitialize:" + name());
        }
        return Result::SUCCESS;
    }

    Tensor output;
};

std::string SyntheticBlockImpl::childStagedValue(const std::string& child) {
    const auto module = moduleHandle(child);
    if (!module) {
        return {};
    }
    const auto* childImpl = module->getImpl<LifecycleChildModule>();
    return childImpl ? childImpl->staged->value : std::string{};
}

std::string SyntheticBlockImpl::childCandidateValue(const std::string& child) {
    const auto module = moduleHandle(child);
    if (!module) {
        return {};
    }
    const auto* childImpl = module->getImpl<LifecycleChildModule>();
    return childImpl ? childImpl->candidate->value : std::string{};
}

std::string SyntheticBlockImpl::childConfigValue(const std::string& child) const {
    const auto it = childConfigs.find(child);
    return it == childConfigs.end() ? std::string{} : it->second->value;
}

TensorLink SyntheticBlockImpl::childOutput(const std::string& child) {
    return moduleGetOutput({child, "out"});
}

std::shared_ptr<Module> MakeChildModule(
    const std::shared_ptr<BlockProbe>& probe,
    const std::shared_ptr<Flowgraph::Environment>& environment,
    const std::shared_ptr<Flowgraph::View>& view) {
    const auto impl = std::make_shared<LifecycleChildModule>();
    impl->probe = probe;

    const auto runtimeContext = std::static_pointer_cast<Runtime::Context>(impl);
    const auto schedulerContext = std::static_pointer_cast<Scheduler::Context>(impl);
    const auto context = std::make_shared<Module::Context>(runtimeContext,
                                                           schedulerContext,
                                                           environment,
                                                           view);
    const auto staged = std::make_shared<LifecycleChildConfig>();
    const auto candidate = std::make_shared<LifecycleChildConfig>();
    impl->staged = staged;
    impl->candidate = candidate;

    return std::make_shared<Module>(DeviceType::CPU,
                                    RuntimeType::NATIVE,
                                    kLifecycleProvider,
                                    impl,
                                    context,
                                    staged,
                                    candidate);
}

struct ScopedChildRegistration {
    explicit ScopedChildRegistration(const std::shared_ptr<BlockProbe>& probe) {
        result = Registry::RegisterModule(
            kChildModuleType,
            DeviceType::CPU,
            RuntimeType::NATIVE,
            kLifecycleProvider,
            [probe](const std::shared_ptr<Flowgraph::Environment>& environment,
                    const std::shared_ptr<Flowgraph::View>& view) {
                return MakeChildModule(probe, environment, view);
            });
        registered = result == Result::SUCCESS;
    }

    ~ScopedChildRegistration() {
        if (registered) {
            (void)Registry::UnregisterModule(kChildModuleType,
                                             DeviceType::CPU,
                                             RuntimeType::NATIVE,
                                             kLifecycleProvider);
        }
    }

    ScopedChildRegistration(const ScopedChildRegistration&) = delete;
    ScopedChildRegistration& operator=(const ScopedChildRegistration&) = delete;

    Result result = Result::ERROR;
    bool registered = false;
};

struct SchedulerHarness {
    SchedulerHarness() : scheduler(std::make_shared<Scheduler>(SchedulerType::SYNCHRONOUS)) {
        createResult = scheduler->create(nullptr);
        created = createResult == Result::SUCCESS;
    }

    ~SchedulerHarness() {
        if (created) {
            (void)scheduler->destroy();
        }
    }

    std::shared_ptr<Scheduler> scheduler;
    Result createResult = Result::ERROR;
    bool created = false;
};

struct SyntheticWindow : Render::Window {
    SyntheticWindow() : Render::Window({}) {}

    const Stats& stats() const override {
        return windowStats;
    }

    std::string info() const override {
        return "synthetic-lifecycle-window";
    }

    constexpr DeviceType device() const override {
        return DeviceType::CPU;
    }

 protected:
    Result bindSurface(const std::shared_ptr<Render::Surface>&) override {
        return Result::SUCCESS;
    }

    Result unbindSurface(const std::shared_ptr<Render::Surface>&) override {
        return Result::SUCCESS;
    }

    Result underlyingCreate() override {
        return Result::SUCCESS;
    }

    Result underlyingDestroy() override {
        return Result::SUCCESS;
    }

    Result underlyingBegin() override {
        return Result::SUCCESS;
    }

    Result underlyingEnd() override {
        return Result::SUCCESS;
    }

    Result underlyingSynchronize() override {
        return Result::SUCCESS;
    }

 private:
    Stats windowStats{};
};

Parser::Map ConfigWithValue(const std::string& value) {
    Parser::Map config;
    config["value"] = value;
    return config;
}

TensorMap InputWithTensor(const Tensor& tensor) {
    TensorMap inputs;
    inputs["in"].produced("upstream", "out", tensor);
    return inputs;
}

std::vector<std::string> EventsStartingWith(const std::vector<std::string>& events,
                                            const std::string& prefix) {
    std::vector<std::string> matches;
    for (const auto& event : events) {
        if (event.starts_with(prefix)) {
            matches.push_back(event);
        }
    }
    return matches;
}

}  // namespace

using namespace Jetstream;

TEST_CASE("Block and Module start from inert lifecycle defaults", "[core][lifecycle]") {
    const auto block = MakeBlock();

    REQUIRE(block.block->name().empty());
    REQUIRE(block.block->state() == Block::State::None);
    REQUIRE(block.block->runtime() == RuntimeType::NONE);
    REQUIRE(block.block->provider().empty());
    REQUIRE(block.block->inputs().empty());
    REQUIRE(block.block->outputs().empty());
    REQUIRE(block.block->interface() == nullptr);
    REQUIRE(block.block->modules().empty());
    REQUIRE(block.block->surfaces().empty());
    REQUIRE(block.block->diagnostic().empty());
    REQUIRE(block.block->config().hash() == block.staged->hash());
    REQUIRE(block.block->config().type() == "cyberether_lifecycle_block");
    REQUIRE(block.block->config().domain() == "test");
    REQUIRE(block.block->config().title() == "Lifecycle Block");
    REQUIRE(block.block->config().summary() == "Synthetic lifecycle fixture.");
    REQUIRE(block.block->config().description() == "Exercises generic block orchestration.");
    REQUIRE(block.block->config().nodeSize() == Block::NodeSize::S);

    CHECK(block.block->device() == DeviceType::None);

    const auto module = MakeModule();

    REQUIRE(module.module->name().empty());
    REQUIRE(module.module->device() == DeviceType::CPU);
    REQUIRE(module.module->runtime() == RuntimeType::NATIVE);
    REQUIRE(module.module->provider() == kLifecycleProvider);
    REQUIRE(module.module->taint() == Module::Taint::CLEAN);
    REQUIRE(module.module->inputs().empty());
    REQUIRE(module.module->outputs().empty());
    REQUIRE(module.module->interface() == nullptr);
    REQUIRE(module.module->surface() == nullptr);
    REQUIRE(module.module->config().hash() == module.staged->hash());
    REQUIRE(module.module->config().type() == "cyberether_lifecycle_module");
    REQUIRE(module.module->context() != nullptr);
}

TEST_CASE("Block states expose stable machine and display names", "[core][lifecycle][block][state]") {
    struct StateName {
        Block::State state;
        const char* machine;
        const char* display;
    };

    const std::vector<StateName> states = {
        {Block::State::None, "none", "None"},
        {Block::State::Creating, "creating", "Creating"},
        {Block::State::Created, "created", "Created"},
        {Block::State::Incomplete, "incomplete", "Incomplete"},
        {Block::State::Errored, "errored", "Errored"},
        {Block::State::Destroying, "destroying", "Destroying"},
        {Block::State::Destroyed, "destroyed", "Destroyed"},
    };

    for (const auto& state : states) {
        REQUIRE(std::string(GetBlockStateName(state.state)) == state.machine);
        REQUIRE(std::string(GetBlockStatePrettyName(state.state)) == state.display);
    }

    const auto unknown = static_cast<Block::State>(255);
    REQUIRE(std::string(GetBlockStateName(unknown)) == "unknown");
    REQUIRE(std::string(GetBlockStatePrettyName(unknown)) == "Unknown");
}

TEST_CASE("Lifecycle contexts retain their generic dependencies", "[core][lifecycle][context]") {
    const auto blockContext = MakeBlockContext();
    REQUIRE(blockContext->instance() == nullptr);
    REQUIRE(blockContext->render() == nullptr);
    REQUIRE(blockContext->scheduler() == nullptr);
    REQUIRE(blockContext->environment() == nullptr);
    REQUIRE(blockContext->view() == nullptr);

    const auto module = MakeModule();
    REQUIRE(module.module->context()->runtime() ==
            std::static_pointer_cast<Runtime::Context>(module.impl));
    REQUIRE(module.module->context()->scheduler() ==
            std::static_pointer_cast<Scheduler::Context>(module.impl));
    REQUIRE(module.module->context()->environment() == nullptr);
    REQUIRE(module.module->context()->view() == nullptr);
}

TEST_CASE("Block forwards environment view and render context to child modules",
          "[core][lifecycle][context][module]") {
    auto bundle = MakeBlock();
    ScopedChildRegistration registration(bundle.probe);
    SchedulerHarness scheduler;
    auto environment = std::make_shared<Flowgraph::Environment>(
        std::shared_ptr<Flowgraph::Impl>{});
    auto view = std::make_shared<Flowgraph::View>(std::shared_ptr<Flowgraph::Impl>{});
    auto render = std::make_shared<SyntheticWindow>();
    bundle.probe->children = {"child"};

    REQUIRE(registration.result == Result::SUCCESS);
    REQUIRE(scheduler.createResult == Result::SUCCESS);
    REQUIRE(bundle.block->create("lifecycle-block",
                                 DeviceType::CPU,
                                 RuntimeType::NATIVE,
                                 kLifecycleProvider,
                                 {},
                                 {},
                                 MakeBlockContext(scheduler.scheduler,
                                                  environment,
                                                  view,
                                                  render)) == Result::SUCCESS);
    REQUIRE(bundle.probe->seenEnvironment == environment);
    REQUIRE(bundle.probe->seenView == view);
    REQUIRE(bundle.probe->seenRender == render);
    REQUIRE(bundle.block->destroy() == Result::SUCCESS);
}

TEST_CASE("Block lifecycle hooks observe committed state in order", "[core][lifecycle][block]") {
    auto bundle = MakeBlock();
    bundle.probe->declareOutput = true;
    bundle.probe->produceOutput = true;

    REQUIRE(bundle.block->create("lifecycle-block",
                                 DeviceType::CPU,
                                 RuntimeType::NATIVE,
                                 kLifecycleProvider,
                                 ConfigWithValue("active"),
                                 {},
                                 MakeBlockContext()) == Result::SUCCESS);

    REQUIRE(bundle.probe->events == std::vector<std::string>{
        "block.candidate.deserialize",
        "block.validate:active:initial",
        "block.staged.deserialize",
        "block.configure:active",
        "block.define:active",
        "block.create:active",
    });
    REQUIRE(bundle.probe->hookStates == std::vector<Block::State>{
        Block::State::Creating,
        Block::State::Creating,
        Block::State::Creating,
        Block::State::Creating,
    });
    REQUIRE(bundle.block->state() == Block::State::Created);
    REQUIRE(bundle.block->name() == "lifecycle-block");
    REQUIRE(bundle.block->device() == DeviceType::CPU);
    REQUIRE(bundle.block->runtime() == RuntimeType::NATIVE);
    REQUIRE(bundle.block->provider() == kLifecycleProvider);
    REQUIRE(bundle.staged->value == "active");
    REQUIRE(bundle.block->outputs().contains("out"));
    REQUIRE(bundle.block->interface()->outputs().size() == 1);

    Parser::Map config;
    REQUIRE(bundle.block->config(config) == Result::SUCCESS);
    REQUIRE(std::any_cast<std::string>(config.at("value")) == "active");
    REQUIRE(bundle.probe->events.back() == "block.staged.serialize");
}

TEST_CASE("Module lifecycle hooks observe committed state in order", "[core][lifecycle][module]") {
    auto bundle = MakeModule();
    bundle.probe->declareOutput = true;
    bundle.probe->produceOutput = true;

    REQUIRE(bundle.module->create("lifecycle-module",
                                  ConfigWithValue("active"),
                                  {}) == Result::SUCCESS);
    REQUIRE(bundle.probe->events == std::vector<std::string>{
        "module.candidate.deserialize",
        "module.validate:active:initial",
        "module.staged.deserialize",
        "module.define:active",
        "module.create:active",
    });
    REQUIRE(bundle.probe->identityReady == std::vector<bool>{true, true, true});
    REQUIRE(bundle.staged->value == "active");
    REQUIRE(bundle.module->outputs().contains("out"));
    REQUIRE(bundle.module->interface()->outputs() == std::vector<std::string>{"out"});
    REQUIRE(bundle.module->surface() != nullptr);

    Parser::Map config;
    REQUIRE(bundle.module->config(config) == Result::SUCCESS);
    REQUIRE(std::any_cast<std::string>(config.at("value")) == "active");
    REQUIRE(bundle.probe->events.back() == "module.staged.serialize");

    REQUIRE(bundle.module->destroy() == Result::SUCCESS);
    REQUIRE(bundle.probe->events.back() == "module.destroy:active");
    REQUIRE(bundle.probe->identityReady.back());
}

TEST_CASE("Synthetic module and child outputs retain complete tensor links",
          "[core][lifecycle][interface][output]") {
    SECTION("module-produced output") {
        auto bundle = MakeModule();
        bundle.probe->declareOutput = true;
        bundle.probe->produceOutput = true;

        REQUIRE(bundle.module->create("lifecycle-module", Parser::Map{}, {}) == Result::SUCCESS);
        REQUIRE(bundle.module->outputs().contains("out"));
        const auto& link = bundle.module->outputs().at("out");
        REQUIRE(link.producer.has_value());
        REQUIRE(link.producer->module == "lifecycle-module");
        REQUIRE(link.producer->port == "out");
        REQUIRE_FALSE(link.external.has_value());
        REQUIRE(link.tensor.device() == DeviceType::CPU);
        REQUIRE(link.tensor.shape() == Shape{2});
        REQUIRE(link.tensor.id() == bundle.impl->output.id());
        REQUIRE(link.tensor.data<F32>()[0] == 3.0f);
        REQUIRE(link.tensor.data<F32>()[1] == 5.0f);
        REQUIRE(bundle.module->destroy() == Result::SUCCESS);
    }

    SECTION("child output exposed by a block") {
        auto bundle = MakeBlock();
        ScopedChildRegistration registration(bundle.probe);
        SchedulerHarness scheduler;
        bundle.probe->children = {"source"};
        bundle.probe->produceChildOutput = true;
        bundle.probe->exposeChildOutput = true;

        REQUIRE(registration.result == Result::SUCCESS);
        REQUIRE(scheduler.createResult == Result::SUCCESS);
        REQUIRE(bundle.block->create("lifecycle-block",
                                     DeviceType::CPU,
                                     RuntimeType::NATIVE,
                                     kLifecycleProvider,
                                     {},
                                     {},
                                     MakeBlockContext(scheduler.scheduler)) == Result::SUCCESS);

        REQUIRE(bundle.block->outputs().contains("out"));
        const auto& exposed = bundle.block->outputs().at("out");
        const auto child = bundle.impl->childOutput("source");
        REQUIRE(exposed.producer.has_value());
        REQUIRE(exposed.producer->module == "lifecycle-block-source");
        REQUIRE(exposed.producer->port == "out");
        REQUIRE(exposed.external.has_value());
        REQUIRE(exposed.external->block == "lifecycle-block");
        REQUIRE(exposed.external->port == "out");
        REQUIRE(exposed.tensor.id() == child.tensor.id());
        REQUIRE(exposed.tensor.data<F32>()[0] == 7.0f);
        REQUIRE(exposed.tensor.data<F32>()[1] == 11.0f);
        REQUIRE(bundle.block->destroy() == Result::SUCCESS);
    }
}

TEST_CASE("Block creation enforces lifecycle and context guards", "[core][lifecycle][block][guard]") {
    auto bundle = MakeBlock();

    REQUIRE(bundle.block->destroy() == Result::ERROR);
    REQUIRE(bundle.block->state() == Block::State::None);
    REQUIRE(bundle.probe->events.empty());

    REQUIRE(bundle.block->create("lifecycle-block",
                                 DeviceType::CPU,
                                 RuntimeType::NATIVE,
                                 kLifecycleProvider,
                                 {},
                                 {},
                                 nullptr) == Result::ERROR);
    REQUIRE(bundle.block->state() == Block::State::None);
    REQUIRE(bundle.probe->events.empty());

    REQUIRE(bundle.block->create("lifecycle-block",
                                 DeviceType::CPU,
                                 RuntimeType::NATIVE,
                                 kLifecycleProvider,
                                 {},
                                 {},
                                 MakeBlockContext()) == Result::SUCCESS);
    const auto events = bundle.probe->events;

    REQUIRE(bundle.block->create("duplicate",
                                 DeviceType::CPU,
                                 RuntimeType::NATIVE,
                                 kLifecycleProvider,
                                 {},
                                 {},
                                 MakeBlockContext()) == Result::ERROR);
    REQUIRE(bundle.block->state() == Block::State::Created);
    REQUIRE(bundle.block->name() == "lifecycle-block");
    REQUIRE(bundle.probe->events == events);
}

TEST_CASE("Block input validation distinguishes incomplete and invalid wiring",
          "[core][lifecycle][block][interface][error]") {
    auto bundle = MakeBlock();

    SECTION("missing declared input is incomplete") {
        bundle.probe->declareInput = true;

        REQUIRE(bundle.block->create("lifecycle-block",
                                     DeviceType::CPU,
                                     RuntimeType::NATIVE,
                                     kLifecycleProvider,
                                     {},
                                     {},
                                     MakeBlockContext()) == Result::INCOMPLETE);
        REQUIRE(bundle.block->state() == Block::State::Incomplete);
        REQUIRE_FALSE(bundle.block->diagnostic().empty());
        REQUIRE(bundle.probe->events.back() == "block.define:initial");
    }

    SECTION("unresolved declared input is incomplete") {
        bundle.probe->declareInput = true;
        TensorMap inputs;
        inputs["in"].requested("upstream", "out");

        REQUIRE(bundle.block->create("lifecycle-block",
                                     DeviceType::CPU,
                                     RuntimeType::NATIVE,
                                     kLifecycleProvider,
                                     {},
                                     inputs,
                                     MakeBlockContext()) == Result::INCOMPLETE);
        REQUIRE(bundle.block->state() == Block::State::Incomplete);
        REQUIRE_FALSE(bundle.block->diagnostic().empty());
        REQUIRE(bundle.probe->events.back() == "block.define:initial");
    }

    SECTION("undeclared input is an error") {
        TensorMap inputs;
        (void)inputs["extra"];

        REQUIRE(bundle.block->create("lifecycle-block",
                                     DeviceType::CPU,
                                     RuntimeType::NATIVE,
                                     kLifecycleProvider,
                                     {},
                                     inputs,
                                     MakeBlockContext()) == Result::ERROR);
        REQUIRE(bundle.block->state() == Block::State::Errored);
        REQUIRE_FALSE(bundle.block->diagnostic().empty());
        REQUIRE(bundle.probe->events.back() == "block.define:initial");
    }

    SECTION("successful recreation clears the previous diagnostic") {
        bundle.probe->declareInput = true;
        REQUIRE(bundle.block->create("lifecycle-block",
                                     DeviceType::CPU,
                                     RuntimeType::NATIVE,
                                     kLifecycleProvider,
                                     {},
                                     {},
                                     MakeBlockContext()) == Result::INCOMPLETE);
        REQUIRE_FALSE(bundle.block->diagnostic().empty());
        const auto diagnostic = bundle.block->diagnostic();
        REQUIRE(bundle.block->destroy() == Result::SUCCESS);

        REQUIRE(bundle.block->create("lifecycle-block",
                                     DeviceType::CPU,
                                     RuntimeType::NATIVE,
                                     kLifecycleProvider,
                                     {},
                                     {},
                                     nullptr) == Result::ERROR);
        REQUIRE(bundle.block->diagnostic() == diagnostic);

        bundle.probe->declareInput = false;
        REQUIRE(bundle.block->create("lifecycle-block",
                                     DeviceType::CPU,
                                     RuntimeType::NATIVE,
                                     kLifecycleProvider,
                                     {},
                                     {},
                                     MakeBlockContext()) == Result::SUCCESS);

        REQUIRE(bundle.block->diagnostic().empty());
    }
}

TEST_CASE("Block creation failures stop orchestration and set state", "[core][lifecycle][block][failure]") {
    auto bundle = MakeBlock();

    SECTION("candidate deserialization failure") {
        bundle.probe->candidateDeserializeResult = Result::ERROR;

        REQUIRE(bundle.block->create("lifecycle-block",
                                     DeviceType::CPU,
                                     RuntimeType::NATIVE,
                                     kLifecycleProvider,
                                     ConfigWithValue("active"),
                                     {},
                                     MakeBlockContext()) == Result::ERROR);
        REQUIRE(bundle.probe->events == std::vector<std::string>{"block.candidate.deserialize"});
        REQUIRE(bundle.block->state() == Block::State::Errored);
        REQUIRE(bundle.staged->value == kInitialValue);
    }

    SECTION("validation failure") {
        bundle.probe->validateResult = Result::ERROR;

        REQUIRE(bundle.block->create("lifecycle-block",
                                     DeviceType::CPU,
                                     RuntimeType::NATIVE,
                                     kLifecycleProvider,
                                     ConfigWithValue("active"),
                                     {},
                                     MakeBlockContext()) == Result::ERROR);
        REQUIRE(bundle.probe->events == std::vector<std::string>{
            "block.candidate.deserialize",
            "block.validate:active:initial",
        });
        REQUIRE(bundle.block->state() == Block::State::Errored);
        REQUIRE(bundle.staged->value == kInitialValue);
    }

    SECTION("staged deserialization failure") {
        bundle.probe->stagedDeserializeResult = Result::ERROR;

        REQUIRE(bundle.block->create("lifecycle-block",
                                     DeviceType::CPU,
                                     RuntimeType::NATIVE,
                                     kLifecycleProvider,
                                     ConfigWithValue("active"),
                                     {},
                                     MakeBlockContext()) == Result::ERROR);
        REQUIRE(bundle.probe->events == std::vector<std::string>{
            "block.candidate.deserialize",
            "block.validate:active:initial",
            "block.staged.deserialize",
        });
        REQUIRE(bundle.block->state() == Block::State::Errored);
        REQUIRE(bundle.staged->value == kInitialValue);
    }

    SECTION("configuration failure") {
        bundle.probe->configureResult = Result::ERROR;

        REQUIRE(bundle.block->create("lifecycle-block",
                                     DeviceType::CPU,
                                     RuntimeType::NATIVE,
                                     kLifecycleProvider,
                                     ConfigWithValue("active"),
                                     {},
                                     MakeBlockContext()) == Result::ERROR);
        REQUIRE(bundle.probe->events == std::vector<std::string>{
            "block.candidate.deserialize",
            "block.validate:active:initial",
            "block.staged.deserialize",
            "block.configure:active",
        });
        REQUIRE(bundle.block->state() == Block::State::Errored);
    }

    SECTION("definition failure") {
        bundle.probe->defineResult = Result::ERROR;

        REQUIRE(bundle.block->create("lifecycle-block",
                                     DeviceType::CPU,
                                     RuntimeType::NATIVE,
                                     kLifecycleProvider,
                                     ConfigWithValue("active"),
                                     {},
                                     MakeBlockContext()) == Result::ERROR);
        REQUIRE(bundle.probe->events.back() == "block.define:active");
        REQUIRE(bundle.block->state() == Block::State::Errored);
        REQUIRE(bundle.block->interface() != nullptr);
    }

    SECTION("create hook failure") {
        bundle.probe->createResult = Result::ERROR;

        REQUIRE(bundle.block->create("lifecycle-block",
                                     DeviceType::CPU,
                                     RuntimeType::NATIVE,
                                     kLifecycleProvider,
                                     ConfigWithValue("active"),
                                     {},
                                     MakeBlockContext()) == Result::ERROR);
        REQUIRE(bundle.probe->events.back() == "block.create:active");
        REQUIRE(bundle.block->state() == Block::State::Errored);
    }

    SECTION("declared output is required") {
        bundle.probe->declareOutput = true;

        REQUIRE(bundle.block->create("lifecycle-block",
                                     DeviceType::CPU,
                                     RuntimeType::NATIVE,
                                     kLifecycleProvider,
                                     ConfigWithValue("active"),
                                     {},
                                     MakeBlockContext()) == Result::ERROR);
        REQUIRE(bundle.probe->events.back() == "block.create:active");
        REQUIRE(bundle.block->state() == Block::State::Errored);
        REQUIRE(bundle.block->outputs().empty());
    }
}

TEST_CASE("Block initial creation accepts recreate validation",
          "[core][lifecycle][block][result]") {
    auto bundle = MakeBlock();
    bundle.probe->validateResult = Result::RECREATE;

    REQUIRE(bundle.block->create("lifecycle-block",
                                 DeviceType::CPU,
                                 RuntimeType::NATIVE,
                                 kLifecycleProvider,
                                 ConfigWithValue("active"),
                                 {},
                                 MakeBlockContext()) == Result::SUCCESS);
    REQUIRE(bundle.block->state() == Block::State::Created);
    REQUIRE(bundle.staged->value == "active");
    REQUIRE(bundle.probe->events.back() == "block.create:active");
}

TEST_CASE("Module creation failures stop orchestration", "[core][lifecycle][module][failure]") {
    auto bundle = MakeModule();

    SECTION("candidate deserialization failure") {
        bundle.probe->candidateDeserializeResult = Result::ERROR;

        REQUIRE(bundle.module->create("lifecycle-module",
                                      ConfigWithValue("active"),
                                      {}) == Result::ERROR);
        REQUIRE(bundle.probe->events == std::vector<std::string>{"module.candidate.deserialize"});
        REQUIRE(bundle.staged->value == kInitialValue);
    }

    SECTION("validation failure") {
        bundle.probe->validateResult = Result::ERROR;

        REQUIRE(bundle.module->create("lifecycle-module",
                                      ConfigWithValue("active"),
                                      {}) == Result::ERROR);
        REQUIRE(bundle.probe->events == std::vector<std::string>{
            "module.candidate.deserialize",
            "module.validate:active:initial",
        });
        REQUIRE(bundle.staged->value == kInitialValue);
    }

    SECTION("staged deserialization failure") {
        bundle.probe->stagedDeserializeResult = Result::ERROR;

        REQUIRE(bundle.module->create("lifecycle-module",
                                      ConfigWithValue("active"),
                                      {}) == Result::ERROR);
        REQUIRE(bundle.probe->events.back() == "module.staged.deserialize");
        REQUIRE(bundle.staged->value == kInitialValue);
    }

    SECTION("definition failure") {
        bundle.probe->defineResult = Result::ERROR;

        REQUIRE(bundle.module->create("lifecycle-module",
                                      ConfigWithValue("active"),
                                      {}) == Result::ERROR);
        REQUIRE(bundle.probe->events.back() == "module.define:active");
    }

    SECTION("create hook failure") {
        bundle.probe->createResult = Result::ERROR;

        REQUIRE(bundle.module->create("lifecycle-module",
                                       ConfigWithValue("active"),
                                       {}) == Result::ERROR);
        REQUIRE(bundle.probe->events.back() == "module.create:active");

        REQUIRE(bundle.module->destroy() == Result::SUCCESS);
        REQUIRE(bundle.probe->events.back() == "module.destroy:active");
    }

    SECTION("declared output is required") {
        bundle.probe->declareOutput = true;

        REQUIRE(bundle.module->create("lifecycle-module",
                                      ConfigWithValue("active"),
                                      {}) == Result::ERROR);
        REQUIRE(bundle.probe->events.back() == "module.create:active");
        REQUIRE(bundle.module->outputs().empty());
    }

    SECTION("missing declared input stops before the create hook") {
        bundle.probe->declareInput = true;

        REQUIRE(bundle.module->create("lifecycle-module",
                                      ConfigWithValue("active"),
                                      {}) == Result::ERROR);
        REQUIRE(bundle.probe->events.back() == "module.define:active");
        REQUIRE(bundle.module->interface()->inputs() == std::vector<std::string>{"in"});
    }
}

TEST_CASE("Module validates real CPU tensor inputs and explicit taints",
          "[core][lifecycle][module][input]") {
    SECTION("valid contiguous input reaches the create hook") {
        auto bundle = MakeModule();
        bundle.probe->declareInput = true;
        Tensor tensor(DeviceType::CPU, DataType::F32, {2, 3});

        REQUIRE(bundle.module->create("lifecycle-module",
                                      Parser::Map{},
                                      InputWithTensor(tensor)) == Result::SUCCESS);
        REQUIRE(bundle.probe->events.back() == "module.create:initial");
        REQUIRE(bundle.module->inputs().at("in").tensor.id() == tensor.id());
        REQUIRE(bundle.module->destroy() == Result::SUCCESS);
    }

    SECTION("CPU input is rejected by a different-device module") {
        auto bundle = MakeModule(DeviceType::CUDA);
        bundle.probe->declareInput = true;
        Tensor tensor(DeviceType::CPU, DataType::F32, {2, 3});

        REQUIRE(bundle.module->create("lifecycle-module",
                                      Parser::Map{},
                                      InputWithTensor(tensor)) == Result::ERROR);
        REQUIRE(bundle.probe->events.back() == "module.define:initial");
    }

    SECTION("cross-device taint admits a real CPU input") {
        auto bundle = MakeModule(DeviceType::CUDA);
        bundle.probe->declareInput = true;
        bundle.probe->taint = Module::Taint::CROSS_DEVICE;
        Tensor tensor(DeviceType::CPU, DataType::F32, {2, 3});

        REQUIRE(bundle.module->create("lifecycle-module",
                                      Parser::Map{},
                                      InputWithTensor(tensor)) == Result::SUCCESS);
        REQUIRE(bundle.module->taint() == Module::Taint::CROSS_DEVICE);
        REQUIRE(bundle.probe->events.back() == "module.create:initial");
        REQUIRE(bundle.module->destroy() == Result::SUCCESS);
    }

    SECTION("invalid CPU shape stops before size validation and create") {
        auto bundle = MakeModule();
        bundle.probe->declareInput = true;
        Tensor tensor(DeviceType::CPU, DataType::F32, {1});
        REQUIRE(tensor.squeezeDims(0) == Result::SUCCESS);
        REQUIRE(tensor.device() == DeviceType::CPU);
        REQUIRE_FALSE(tensor.validShape());

        REQUIRE(bundle.module->create("lifecycle-module",
                                      Parser::Map{},
                                      InputWithTensor(tensor)) == Result::ERROR);
        REQUIRE(bundle.probe->events.back() == "module.define:initial");
    }

    SECTION("zero-sized CPU input is rejected") {
        auto bundle = MakeModule();
        bundle.probe->declareInput = true;
        Tensor tensor(DeviceType::CPU, DataType::F32, {2, 0, 3});
        REQUIRE(tensor.validShape());
        REQUIRE(tensor.size() == 0);

        REQUIRE(bundle.module->create("lifecycle-module",
                                      Parser::Map{},
                                      InputWithTensor(tensor)) == Result::ERROR);
        REQUIRE(bundle.probe->events.back() == "module.define:initial");
    }

    SECTION("non-contiguous CPU input is rejected without a taint") {
        auto bundle = MakeModule();
        bundle.probe->declareInput = true;
        Tensor tensor(DeviceType::CPU, DataType::F32, {2, 3});
        REQUIRE(tensor.permute({1, 0}) == Result::SUCCESS);
        REQUIRE_FALSE(tensor.contiguous());

        REQUIRE(bundle.module->create("lifecycle-module",
                                      Parser::Map{},
                                      InputWithTensor(tensor)) == Result::ERROR);
        REQUIRE(bundle.probe->events.back() == "module.define:initial");
    }

    SECTION("discontiguous taint admits a non-contiguous CPU input") {
        auto bundle = MakeModule();
        bundle.probe->declareInput = true;
        bundle.probe->taint = Module::Taint::DISCONTIGUOUS;
        Tensor tensor(DeviceType::CPU, DataType::F32, {2, 3});
        REQUIRE(tensor.permute({1, 0}) == Result::SUCCESS);

        REQUIRE(bundle.module->create("lifecycle-module",
                                      Parser::Map{},
                                      InputWithTensor(tensor)) == Result::SUCCESS);
        REQUIRE(bundle.module->taint() == Module::Taint::DISCONTIGUOUS);
        REQUIRE(bundle.probe->events.back() == "module.create:initial");
        REQUIRE(bundle.module->destroy() == Result::SUCCESS);
    }
}

TEST_CASE("Module post-create failures invoke deterministic cleanup",
          "[core][lifecycle][module][cleanup][failure]") {
    SECTION("create hook failure invokes destroy") {
        auto bundle = MakeModule();
        bundle.probe->createResult = Result::ERROR;

        REQUIRE(bundle.module->create("lifecycle-module", Parser::Map{}, {}) == Result::ERROR);

        // Expected failure: Module::create returns without invoking destroy after hook failure.
        CHECK(bundle.probe->events.back() == "module.destroy:initial");
        if (bundle.probe->events.back() != "module.destroy:initial") {
            REQUIRE(bundle.module->destroy() == Result::SUCCESS);
        }
    }

    SECTION("output validation failure invokes destroy") {
        auto bundle = MakeModule();
        bundle.probe->declareOutput = true;

        REQUIRE(bundle.module->create("lifecycle-module", Parser::Map{}, {}) == Result::ERROR);

        // Expected failure: Module::create does not clean up after output validation fails.
        CHECK(bundle.probe->events.back() == "module.destroy:initial");
        if (bundle.probe->events.back() != "module.destroy:initial") {
            REQUIRE(bundle.module->destroy() == Result::SUCCESS);
        }
    }

    SECTION("incomplete create remains explicitly destroyable") {
        auto bundle = MakeModule();
        bundle.probe->createResult = Result::INCOMPLETE;

        REQUIRE(bundle.module->create("lifecycle-module", Parser::Map{}, {}) ==
                Result::INCOMPLETE);
        REQUIRE(bundle.probe->events.back() == "module.create:initial");
        REQUIRE(bundle.module->destroy() == Result::SUCCESS);
        REQUIRE(bundle.probe->events.back() == "module.destroy:initial");
    }
}

TEST_CASE("Module configuration sources propagate serialization results",
          "[core][lifecycle][module][configuration]") {
    auto bundle = MakeModule();
    SyntheticModuleConfig config;
    config.probe = bundle.probe;
    config.role = ConfigRole::Candidate;
    config.value = "active";

    SECTION("configuration object is serialized before creation") {
        REQUIRE(bundle.module->create("lifecycle-module", config, {}) == Result::SUCCESS);
        REQUIRE(bundle.probe->events == std::vector<std::string>{
            "module.candidate.serialize",
            "module.candidate.deserialize",
            "module.validate:active:initial",
            "module.staged.deserialize",
            "module.define:active",
            "module.create:active",
        });
        REQUIRE(bundle.staged->value == "active");
    }

    SECTION("configuration object serialization failure stops before mutation") {
        config.serializeResult = Result::ERROR;

        REQUIRE(bundle.module->create("lifecycle-module", config, {}) == Result::ERROR);
        REQUIRE(bundle.probe->events == std::vector<std::string>{"module.candidate.serialize"});
        REQUIRE(bundle.module->name().empty());
        REQUIRE(bundle.module->interface() == nullptr);
        REQUIRE(bundle.staged->value == kInitialValue);
    }

    SECTION("stored configuration serialization failure is returned") {
        REQUIRE(bundle.module->create("lifecycle-module",
                                      ConfigWithValue("active"),
                                      {}) == Result::SUCCESS);
        bundle.probe->events.clear();
        bundle.staged->serializeResult = Result::ERROR;

        Parser::Map serialized;
        REQUIRE(bundle.module->config(serialized) == Result::ERROR);
        REQUIRE(serialized.empty());
        REQUIRE(bundle.probe->events == std::vector<std::string>{"module.staged.serialize"});
    }
}

TEST_CASE("Module destroy propagates implementation errors",
          "[core][lifecycle][module][cleanup]") {
    auto bundle = MakeModule();

    REQUIRE(bundle.module->create("lifecycle-module", Parser::Map{}, {}) == Result::SUCCESS);
    bundle.probe->events.clear();
    bundle.probe->identityReady.clear();
    bundle.probe->destroyResult = Result::ERROR;

    REQUIRE(bundle.module->destroy() == Result::ERROR);
    REQUIRE(bundle.probe->events == std::vector<std::string>{"module.destroy:initial"});
    REQUIRE(bundle.probe->identityReady == std::vector<bool>{true});

    bundle.probe->destroyResult = Result::SUCCESS;
    REQUIRE(bundle.module->destroy() == Result::SUCCESS);
    REQUIRE(bundle.probe->events.back() == "module.destroy:initial");
}

TEST_CASE("Module repeated lifecycle calls preserve explicit state semantics",
          "[core][lifecycle][module][state]") {
    // TODO: Assert a concrete lifecycle state once Module exposes a state machine or accessor.
    SECTION("destroy before create is rejected without calling the hook") {
        auto bundle = MakeModule();

        const auto result = bundle.module->destroy();

        // Expected failure: Module has no guard against destruction before creation.
        CHECK(result == Result::ERROR);
        // Expected failure: uncreated Module destruction still invokes the implementation hook.
        CHECK(bundle.probe->events.empty());
    }

    SECTION("a second create is rejected without replacing live state") {
        auto bundle = MakeModule();
        REQUIRE(bundle.module->create("lifecycle-module",
                                      ConfigWithValue("before"),
                                      {}) == Result::SUCCESS);
        bundle.probe->events.clear();

        const auto result = bundle.module->create("replacement",
                                                   ConfigWithValue("after"),
                                                   {});

        // Expected failure: Module::create permits creation over a live module.
        CHECK(result == Result::ERROR);
        // Expected failure: repeated creation reruns lifecycle hooks.
        CHECK(bundle.probe->events.empty());
        // Expected failure: repeated creation replaces the live module identity.
        CHECK(bundle.module->name() == "lifecycle-module");
        // Expected failure: repeated creation commits the replacement configuration.
        CHECK(bundle.staged->value == "before");
        REQUIRE(bundle.module->destroy() == Result::SUCCESS);
    }

    SECTION("a second destroy is rejected without calling the hook") {
        auto bundle = MakeModule();
        REQUIRE(bundle.module->create("lifecycle-module", Parser::Map{}, {}) == Result::SUCCESS);
        REQUIRE(bundle.module->destroy() == Result::SUCCESS);
        bundle.probe->events.clear();

        const auto result = bundle.module->destroy();

        // Expected failure: Module has no guard against repeated destruction.
        CHECK(result == Result::ERROR);
        // Expected failure: repeated destruction reruns the implementation hook.
        CHECK(bundle.probe->events.empty());
    }

    SECTION("reconfigure before create is rejected without committing") {
        auto bundle = MakeModule();

        const auto result = bundle.module->reconfigure(ConfigWithValue("after"));

        // Expected failure: Module reconfigures before it has been created.
        CHECK(result == Result::ERROR);
        // Expected failure: pre-create reconfiguration commits staged state.
        CHECK(bundle.staged->value == kInitialValue);
        // Expected failure: pre-create reconfiguration invokes lifecycle hooks.
        CHECK(bundle.probe->events.empty());
    }

    SECTION("reconfigure after destroy requires recreation") {
        auto bundle = MakeModule();
        REQUIRE(bundle.module->create("lifecycle-module",
                                      ConfigWithValue("before"),
                                      {}) == Result::SUCCESS);
        REQUIRE(bundle.module->destroy() == Result::SUCCESS);
        bundle.probe->events.clear();

        const auto result = bundle.module->reconfigure(ConfigWithValue("after"));

        // Expected failure: destroyed Module reconfiguration is not rejected as recreation.
        CHECK(result == Result::RECREATE);
        // Expected failure: destroyed Module reconfiguration commits staged state.
        CHECK(bundle.staged->value == "before");
        // Expected failure: destroyed Module reconfiguration invokes lifecycle hooks.
        CHECK(bundle.probe->events.empty());
    }

    SECTION("create after destroy starts a fresh lifecycle") {
        auto bundle = MakeModule();
        REQUIRE(bundle.module->create("lifecycle-module",
                                      ConfigWithValue("before"),
                                      {}) == Result::SUCCESS);
        REQUIRE(bundle.module->destroy() == Result::SUCCESS);
        bundle.probe->events.clear();

        REQUIRE(bundle.module->create("lifecycle-module",
                                      ConfigWithValue("after"),
                                      {}) == Result::SUCCESS);
        REQUIRE(bundle.staged->value == "after");
        REQUIRE(bundle.probe->events.back() == "module.create:after");
        REQUIRE(bundle.module->destroy() == Result::SUCCESS);
    }
}

TEST_CASE("Block interface declarations reject duplicate keys", "[core][lifecycle][interface]") {
    auto bundle = MakeBlock();

    SECTION("input") {
        bundle.probe->duplicate = DuplicateInterface::Input;
        REQUIRE(bundle.block->create("lifecycle-block",
                                     DeviceType::CPU,
                                     RuntimeType::NATIVE,
                                     kLifecycleProvider,
                                     {},
                                     {},
                                     MakeBlockContext()) == Result::ERROR);
        REQUIRE(bundle.block->interface()->inputs().size() == 1);
    }

    SECTION("output") {
        bundle.probe->duplicate = DuplicateInterface::Output;
        REQUIRE(bundle.block->create("lifecycle-block",
                                     DeviceType::CPU,
                                     RuntimeType::NATIVE,
                                     kLifecycleProvider,
                                     {},
                                     {},
                                     MakeBlockContext()) == Result::ERROR);
        REQUIRE(bundle.block->interface()->outputs().size() == 1);
    }

    SECTION("config") {
        bundle.probe->duplicate = DuplicateInterface::Config;
        REQUIRE(bundle.block->create("lifecycle-block",
                                     DeviceType::CPU,
                                     RuntimeType::NATIVE,
                                     kLifecycleProvider,
                                     {},
                                     {},
                                     MakeBlockContext()) == Result::ERROR);
        REQUIRE(bundle.block->interface()->configs().size() == 1);
    }

    SECTION("metric") {
        bundle.probe->duplicate = DuplicateInterface::Metric;
        REQUIRE(bundle.block->create("lifecycle-block",
                                     DeviceType::CPU,
                                     RuntimeType::NATIVE,
                                     kLifecycleProvider,
                                     {},
                                     {},
                                     MakeBlockContext()) == Result::ERROR);
        REQUIRE(bundle.block->interface()->metrics().size() == 1);
    }

    REQUIRE(bundle.block->state() == Block::State::Errored);
}

TEST_CASE("Module interface declarations reject duplicate keys", "[core][lifecycle][interface]") {
    auto bundle = MakeModule();

    SECTION("input") {
        bundle.probe->duplicate = DuplicateInterface::Input;
        REQUIRE(bundle.module->create("lifecycle-module", Parser::Map{}, {}) == Result::ERROR);
        REQUIRE(bundle.module->interface()->inputs() == std::vector<std::string>{"port"});
    }

    SECTION("output") {
        bundle.probe->duplicate = DuplicateInterface::Output;
        REQUIRE(bundle.module->create("lifecycle-module", Parser::Map{}, {}) == Result::ERROR);
        REQUIRE(bundle.module->interface()->outputs() == std::vector<std::string>{"port"});
    }
}

TEST_CASE("Block interfaces retain declaration metadata and metric callbacks",
          "[core][lifecycle][block][interface]") {
    auto bundle = MakeBlock();
    bundle.probe->declareInput = true;
    bundle.probe->declareOutput = true;
    bundle.probe->declareConfig = true;
    bundle.probe->declareMetric = true;

    REQUIRE(bundle.block->create("lifecycle-block",
                                 DeviceType::CPU,
                                 RuntimeType::NATIVE,
                                 kLifecycleProvider,
                                 {},
                                 {},
                                 MakeBlockContext()) == Result::INCOMPLETE);

    const auto interface = bundle.block->interface();
    REQUIRE(interface != nullptr);
    REQUIRE(interface->inputs().size() == 1);
    REQUIRE(interface->outputs().size() == 1);
    REQUIRE(interface->configs().size() == 1);
    REQUIRE(interface->metrics().size() == 1);

    REQUIRE(interface->inputs()[0].first == "in");
    REQUIRE(interface->inputs()[0].second.label == "Input");
    REQUIRE(interface->inputs()[0].second.format.empty());
    REQUIRE(interface->inputs()[0].second.help == "Synthetic input.");
    REQUIRE_FALSE(static_cast<bool>(interface->inputs()[0].second.metric));

    REQUIRE(interface->outputs()[0].first == "out");
    REQUIRE(interface->outputs()[0].second.label == "Output");
    REQUIRE(interface->outputs()[0].second.format.empty());
    REQUIRE(interface->outputs()[0].second.help == "Synthetic output.");
    REQUIRE_FALSE(static_cast<bool>(interface->outputs()[0].second.metric));

    REQUIRE(interface->configs()[0].first == "value");
    REQUIRE(interface->configs()[0].second.label == "Value");
    REQUIRE(interface->configs()[0].second.format == "text");
    REQUIRE(interface->configs()[0].second.help == "Synthetic config.");
    REQUIRE_FALSE(static_cast<bool>(interface->configs()[0].second.metric));

    REQUIRE(interface->metrics()[0].first == "status");
    REQUIRE(interface->metrics()[0].second.label == "Status");
    REQUIRE(interface->metrics()[0].second.format == "text");
    REQUIRE(interface->metrics()[0].second.help == "Synthetic metric.");
    REQUIRE(static_cast<bool>(interface->metrics()[0].second.metric));
    REQUIRE(std::any_cast<std::string>(interface->metrics()[0].second.metric()) == "ready");
}

TEST_CASE("Block cleans child modules in reverse creation order", "[core][lifecycle][cleanup]") {
    auto bundle = MakeBlock();
    ScopedChildRegistration registration(bundle.probe);
    SchedulerHarness scheduler;

    REQUIRE(registration.result == Result::SUCCESS);
    REQUIRE(scheduler.createResult == Result::SUCCESS);
    bundle.probe->children = {"first", "second"};

    SECTION("create failure cleans all created children") {
        bundle.probe->createResult = Result::ERROR;

        REQUIRE(bundle.block->create("lifecycle-block",
                                     DeviceType::CPU,
                                     RuntimeType::NATIVE,
                                     kLifecycleProvider,
                                     {},
                                     {},
                                     MakeBlockContext(scheduler.scheduler)) == Result::ERROR);
        REQUIRE(bundle.block->state() == Block::State::Errored);
        REQUIRE(bundle.block->modules().empty());
        REQUIRE(bundle.probe->events[bundle.probe->events.size() - 2] ==
                "child.destroy:lifecycle-block-second");
        REQUIRE(bundle.probe->events.back() == "child.destroy:lifecycle-block-first");
    }

    SECTION("output validation failure cleans all created children") {
        bundle.probe->declareOutput = true;

        REQUIRE(bundle.block->create("lifecycle-block",
                                     DeviceType::CPU,
                                     RuntimeType::NATIVE,
                                     kLifecycleProvider,
                                     {},
                                     {},
                                     MakeBlockContext(scheduler.scheduler)) == Result::ERROR);
        REQUIRE(bundle.block->state() == Block::State::Errored);
        REQUIRE(bundle.block->modules().empty());
        REQUIRE(bundle.probe->events[bundle.probe->events.size() - 2] ==
                "child.destroy:lifecycle-block-second");
        REQUIRE(bundle.probe->events.back() == "child.destroy:lifecycle-block-first");
    }

    SECTION("child creation failure cleans the failed child and prior children") {
        bundle.probe->failingChildCreate = "second";

        REQUIRE(bundle.block->create("lifecycle-block",
                                     DeviceType::CPU,
                                     RuntimeType::NATIVE,
                                     kLifecycleProvider,
                                     {},
                                     {},
                                     MakeBlockContext(scheduler.scheduler)) == Result::ERROR);
        REQUIRE(bundle.block->state() == Block::State::Errored);
        REQUIRE(bundle.block->modules().empty());
        REQUIRE(bundle.probe->events[bundle.probe->events.size() - 2] ==
                "child.destroy:lifecycle-block-second");
        REQUIRE(bundle.probe->events.back() == "child.destroy:lifecycle-block-first");
    }

    SECTION("failure cleanup continues after a child destroy error") {
        bundle.probe->createResult = Result::ERROR;
        bundle.probe->failingChildDestroy = "second";

        REQUIRE(bundle.block->create("lifecycle-block",
                                     DeviceType::CPU,
                                     RuntimeType::NATIVE,
                                     kLifecycleProvider,
                                     {},
                                     {},
                                     MakeBlockContext(scheduler.scheduler)) == Result::ERROR);
        REQUIRE(bundle.block->state() == Block::State::Errored);
        REQUIRE(bundle.block->modules().empty());
        REQUIRE(bundle.probe->events[bundle.probe->events.size() - 2] ==
                "child.destroy:lifecycle-block-second");
        REQUIRE(bundle.probe->events.back() == "child.destroy:lifecycle-block-first");
    }

    SECTION("explicit destruction can resume after a child destroy error") {
        REQUIRE(bundle.block->create("lifecycle-block",
                                     DeviceType::CPU,
                                     RuntimeType::NATIVE,
                                     kLifecycleProvider,
                                     {},
                                     {},
                                     MakeBlockContext(scheduler.scheduler)) == Result::SUCCESS);
        bundle.probe->events.clear();
        bundle.probe->failingChildDestroy = "second";

        REQUIRE(bundle.block->destroy() == Result::ERROR);
        REQUIRE(bundle.block->state() == Block::State::Errored);
        REQUIRE(bundle.block->modules() == std::vector<std::string>{"first"});
        REQUIRE(bundle.probe->events ==
                std::vector<std::string>{"child.destroy:lifecycle-block-second"});

        bundle.probe->failingChildDestroy.clear();
        REQUIRE(bundle.block->destroy() == Result::SUCCESS);
        REQUIRE(bundle.block->state() == Block::State::Destroyed);
        REQUIRE(bundle.block->modules().empty());
        REQUIRE(bundle.probe->events.back() == "child.destroy:lifecycle-block-first");
    }

    SECTION("explicit destruction runs children before the block hook") {
        REQUIRE(bundle.block->create("lifecycle-block",
                                     DeviceType::CPU,
                                     RuntimeType::NATIVE,
                                     kLifecycleProvider,
                                     {},
                                     {},
                                     MakeBlockContext(scheduler.scheduler)) == Result::SUCCESS);
        REQUIRE(bundle.block->modules() == std::vector<std::string>{"first", "second"});

        bundle.probe->events.clear();
        REQUIRE(bundle.block->destroy() == Result::SUCCESS);
        REQUIRE(bundle.block->state() == Block::State::Destroyed);
        REQUIRE(bundle.block->modules().empty());
        REQUIRE(bundle.probe->events.size() >= 2);
        REQUIRE(bundle.probe->events[0] == "child.destroy:lifecycle-block-second");
        REQUIRE(bundle.probe->events[1] == "child.destroy:lifecycle-block-first");

        // Expected failure: Block::destroy skips the implementation hook after child cleanup.
        REQUIRE(bundle.probe->events == std::vector<std::string>{
            "child.destroy:lifecycle-block-second",
            "child.destroy:lifecycle-block-first",
            "block.destroy",
        });
    }

    SECTION("block destroy hook errors are propagated after child cleanup") {
        REQUIRE(bundle.block->create("lifecycle-block",
                                     DeviceType::CPU,
                                     RuntimeType::NATIVE,
                                     kLifecycleProvider,
                                     {},
                                     {},
                                     MakeBlockContext(scheduler.scheduler)) == Result::SUCCESS);
        bundle.probe->events.clear();
        bundle.probe->destroyResult = Result::ERROR;

        const auto result = bundle.block->destroy();

        // Expected failure: Block::destroy never calls or propagates the implementation hook.
        CHECK(result == Result::ERROR);
        CHECK(bundle.block->state() == Block::State::Errored);
        CHECK(bundle.probe->events == std::vector<std::string>{
            "child.destroy:lifecycle-block-second",
            "child.destroy:lifecycle-block-first",
            "block.destroy",
        });
    }
}

TEST_CASE("Block child creation handles duplicates registry errors and incompleteness",
          "[core][lifecycle][module][failure]") {
    auto bundle = MakeBlock();
    ScopedChildRegistration registration(bundle.probe);
    SchedulerHarness scheduler;

    REQUIRE(registration.result == Result::SUCCESS);
    REQUIRE(scheduler.createResult == Result::SUCCESS);

    SECTION("duplicate child names reject the second and clean the first") {
        bundle.probe->children = {"duplicate", "duplicate"};

        REQUIRE(bundle.block->create("lifecycle-block",
                                     DeviceType::CPU,
                                     RuntimeType::NATIVE,
                                     kLifecycleProvider,
                                     {},
                                     {},
                                     MakeBlockContext(scheduler.scheduler)) == Result::ERROR);
        REQUIRE(bundle.block->state() == Block::State::Errored);
        REQUIRE(bundle.block->modules().empty());
        REQUIRE(std::count(bundle.probe->events.begin(),
                           bundle.probe->events.end(),
                           "child.create:lifecycle-block-duplicate") == 1);
        REQUIRE(std::count(bundle.probe->events.begin(),
                           bundle.probe->events.end(),
                           "child.destroy:lifecycle-block-duplicate") == 1);
    }

    SECTION("missing registry entry stops before child creation") {
        bundle.probe->children = {"missing"};
        bundle.probe->childModuleType = kMissingChildModuleType;

        REQUIRE(bundle.block->create("lifecycle-block",
                                     DeviceType::CPU,
                                     RuntimeType::NATIVE,
                                     kLifecycleProvider,
                                     {},
                                     {},
                                     MakeBlockContext(scheduler.scheduler)) == Result::ERROR);
        REQUIRE(bundle.block->state() == Block::State::Errored);
        REQUIRE(bundle.block->modules().empty());
        REQUIRE(EventsStartingWith(bundle.probe->events, "child.").empty());

        // TODO: A null registry factory product has no safe test seam until moduleCreate
        // validates the built pointer before dereferencing it.
    }

    SECTION("incomplete child is retained for explicit destruction") {
        bundle.probe->children = {"pending"};
        bundle.probe->incompleteChildCreate = "pending";

        REQUIRE(bundle.block->create("lifecycle-block",
                                     DeviceType::CPU,
                                     RuntimeType::NATIVE,
                                     kLifecycleProvider,
                                     {},
                                     {},
                                     MakeBlockContext(scheduler.scheduler)) ==
                Result::INCOMPLETE);
        REQUIRE(bundle.block->state() == Block::State::Incomplete);
        REQUIRE(bundle.block->modules() == std::vector<std::string>{"pending"});
        REQUIRE(std::count(bundle.probe->events.begin(),
                           bundle.probe->events.end(),
                           "child.destroy:lifecycle-block-pending") == 0);

        REQUIRE(bundle.block->destroy() == Result::SUCCESS);
        REQUIRE(bundle.block->state() == Block::State::Destroyed);
        REQUIRE(bundle.block->modules().empty());
        REQUIRE(bundle.probe->events.back() == "child.destroy:lifecycle-block-pending");
    }

    SECTION("incomplete block create retains successful children") {
        bundle.probe->children = {"first", "second"};
        bundle.probe->createResult = Result::INCOMPLETE;

        REQUIRE(bundle.block->create("lifecycle-block",
                                     DeviceType::CPU,
                                     RuntimeType::NATIVE,
                                     kLifecycleProvider,
                                     {},
                                     {},
                                     MakeBlockContext(scheduler.scheduler)) ==
                Result::INCOMPLETE);
        REQUIRE(bundle.block->state() == Block::State::Incomplete);
        REQUIRE(bundle.block->modules() == std::vector<std::string>{"first", "second"});

        REQUIRE(bundle.block->destroy() == Result::SUCCESS);
        REQUIRE(bundle.block->modules().empty());
        REQUIRE(bundle.probe->events[bundle.probe->events.size() - 2] ==
                "child.destroy:lifecycle-block-second");
        REQUIRE(bundle.probe->events.back() == "child.destroy:lifecycle-block-first");
    }

    SECTION("scheduler presentation add error destroys the untracked child") {
        bundle.probe->children = {"rejected"};
        bundle.probe->recordSchedulerEvents = true;
        bundle.probe->failingChildPresentInitialize = "rejected";

        REQUIRE(bundle.block->create("lifecycle-block",
                                     DeviceType::CPU,
                                     RuntimeType::NATIVE,
                                     kLifecycleProvider,
                                     {},
                                     {},
                                     MakeBlockContext(scheduler.scheduler)) == Result::ERROR);
        REQUIRE(bundle.block->state() == Block::State::Errored);
        REQUIRE(bundle.block->modules().empty());
        REQUIRE(EventsStartingWith(bundle.probe->events, "child.") ==
                std::vector<std::string>{
                    "child.create:lifecycle-block-rejected",
                    "child.present_initialize:lifecycle-block-rejected",
                    "child.destroy:lifecycle-block-rejected",
                });
    }

    SECTION("scheduler runtime add error deinitializes and destroys the child") {
        bundle.probe->children = {"rejected"};
        bundle.probe->recordSchedulerEvents = true;
        bundle.probe->failingChildInitialize = "rejected";

        REQUIRE(bundle.block->create("lifecycle-block",
                                     DeviceType::CPU,
                                     RuntimeType::NATIVE,
                                     kLifecycleProvider,
                                     {},
                                     {},
                                     MakeBlockContext(scheduler.scheduler)) == Result::ERROR);
        REQUIRE(bundle.block->state() == Block::State::Errored);
        REQUIRE(bundle.block->modules().empty());
        REQUIRE(EventsStartingWith(bundle.probe->events, "child.") ==
                std::vector<std::string>{
                    "child.create:lifecycle-block-rejected",
                    "child.present_initialize:lifecycle-block-rejected",
                    "child.initialize:lifecycle-block-rejected",
                    "child.deinitialize:lifecycle-block-rejected",
                    "child.destroy:lifecycle-block-rejected",
                });
    }
}

TEST_CASE("Block preserves child ownership across scheduler remove failure",
          "[core][lifecycle][module][cleanup][failure]") {
    auto bundle = MakeBlock();
    ScopedChildRegistration registration(bundle.probe);
    SchedulerHarness scheduler;
    bundle.probe->children = {"first", "second"};

    REQUIRE(registration.result == Result::SUCCESS);
    REQUIRE(scheduler.createResult == Result::SUCCESS);
    REQUIRE(bundle.block->create("lifecycle-block",
                                 DeviceType::CPU,
                                 RuntimeType::NATIVE,
                                 kLifecycleProvider,
                                 {},
                                 {},
                                 MakeBlockContext(scheduler.scheduler)) == Result::SUCCESS);
    bundle.probe->events.clear();
    bundle.probe->recordSchedulerEvents = true;
    bundle.probe->failingChildInitialize = "first";

    REQUIRE(bundle.block->destroy() == Result::ERROR);
    REQUIRE(bundle.block->state() == Block::State::Errored);
    REQUIRE(EventsStartingWith(bundle.probe->events, "child.initialize:") ==
            std::vector<std::string>{"child.initialize:lifecycle-block-first"});

    // Expected failure: moduleDestroy erases the child before scheduler removal succeeds.
    CHECK(bundle.block->modules() == std::vector<std::string>{"first", "second"});

    bundle.probe->failingChildInitialize.clear();
    REQUIRE(bundle.block->destroy() == Result::SUCCESS);
    REQUIRE(bundle.block->modules().empty());
    REQUIRE(std::count(bundle.probe->events.begin(),
                       bundle.probe->events.end(),
                       "child.destroy:lifecycle-block-first") == 1);
    // Expected failure: the child lost on scheduler removal failure cannot be cleaned on retry.
    CHECK(std::count(bundle.probe->events.begin(),
                     bundle.probe->events.end(),
                     "child.destroy:lifecycle-block-second") == 1);
}

TEST_CASE("Block retries child destruction after a partial cleanup failure",
          "[core][lifecycle][module][cleanup][failure]") {
    auto bundle = MakeBlock();
    ScopedChildRegistration registration(bundle.probe);
    SchedulerHarness scheduler;
    bundle.probe->children = {"first", "second"};

    REQUIRE(registration.result == Result::SUCCESS);
    REQUIRE(scheduler.createResult == Result::SUCCESS);
    REQUIRE(bundle.block->create("lifecycle-block",
                                 DeviceType::CPU,
                                 RuntimeType::NATIVE,
                                 kLifecycleProvider,
                                 {},
                                 {},
                                 MakeBlockContext(scheduler.scheduler)) == Result::SUCCESS);
    bundle.probe->events.clear();
    bundle.probe->failingChildDestroy = "second";

    REQUIRE(bundle.block->destroy() == Result::ERROR);
    REQUIRE(bundle.block->state() == Block::State::Errored);

    // Expected failure: moduleDestroy discards ownership of a child whose destroy hook failed.
    CHECK(bundle.block->modules() == std::vector<std::string>{"first", "second"});

    bundle.probe->failingChildDestroy.clear();
    REQUIRE(bundle.block->destroy() == Result::SUCCESS);
    REQUIRE(bundle.block->modules().empty());
    // Expected failure: retry cannot revisit the child erased by the first destroy attempt.
    CHECK(std::count(bundle.probe->events.begin(),
                     bundle.probe->events.end(),
                     "child.destroy:lifecycle-block-second") == 2);
    REQUIRE(std::count(bundle.probe->events.begin(),
                       bundle.probe->events.end(),
                       "child.destroy:lifecycle-block-first") == 1);
}

TEST_CASE("Block reconfiguration is validated and atomic", "[core][lifecycle][reconfigure]") {
    auto bundle = MakeBlock();
    bundle.probe->declareOutput = true;
    bundle.probe->produceOutput = true;

    REQUIRE(bundle.block->create("lifecycle-block",
                                 DeviceType::CPU,
                                 RuntimeType::NATIVE,
                                 kLifecycleProvider,
                                 ConfigWithValue("before"),
                                 {},
                                 MakeBlockContext()) == Result::SUCCESS);
    bundle.probe->events.clear();
    bundle.probe->hookStates.clear();

    SECTION("unchanged configuration stops after candidate deserialization") {
        REQUIRE(bundle.block->reconfigure(ConfigWithValue("before")) == Result::SUCCESS);
        REQUIRE(bundle.probe->events == std::vector<std::string>{"block.candidate.deserialize"});
        REQUIRE(bundle.staged->value == "before");
    }

    SECTION("candidate deserialization failure preserves the staged configuration") {
        bundle.probe->candidateDeserializeResult = Result::ERROR;

        REQUIRE(bundle.block->reconfigure(ConfigWithValue("after")) == Result::ERROR);
        REQUIRE(bundle.probe->events ==
                std::vector<std::string>{"block.candidate.deserialize"});
        REQUIRE(bundle.staged->value == "before");
        REQUIRE(bundle.block->state() == Block::State::Created);
    }

    SECTION("changed configuration validates, commits, configures, and refreshes the interface") {
        REQUIRE(bundle.block->reconfigure(ConfigWithValue("after")) == Result::SUCCESS);
        REQUIRE(bundle.probe->events == std::vector<std::string>{
            "block.candidate.deserialize",
            "block.validate:after:before",
            "block.staged.serialize",
            "block.staged.deserialize",
            "block.configure:after",
            "block.define:after",
        });
        REQUIRE(bundle.probe->hookStates == std::vector<Block::State>{
            Block::State::Created,
            Block::State::Created,
            Block::State::Created,
        });
        REQUIRE(bundle.staged->value == "after");
        REQUIRE(bundle.block->interface()->outputs().size() == 1);
        REQUIRE(bundle.block->state() == Block::State::Created);
    }

    SECTION("validation failure preserves the staged configuration") {
        bundle.probe->validateResult = Result::ERROR;

        REQUIRE(bundle.block->reconfigure(ConfigWithValue("after")) == Result::ERROR);
        REQUIRE(bundle.probe->events == std::vector<std::string>{
            "block.candidate.deserialize",
            "block.validate:after:before",
        });
        REQUIRE(bundle.staged->value == "before");
        REQUIRE(bundle.block->state() == Block::State::Created);
    }

    SECTION("recreate validation result is propagated without committing") {
        bundle.probe->validateResult = Result::RECREATE;

        REQUIRE(bundle.block->reconfigure(ConfigWithValue("after")) == Result::RECREATE);
        REQUIRE(bundle.staged->value == "before");
        REQUIRE(bundle.block->state() == Block::State::Created);
    }

    SECTION("configuration backup serialization failure stops before commit") {
        bundle.staged->serializeResult = Result::ERROR;

        REQUIRE(bundle.block->reconfigure(ConfigWithValue("after")) == Result::ERROR);
        REQUIRE(bundle.probe->events == std::vector<std::string>{
            "block.candidate.deserialize",
            "block.validate:after:before",
            "block.staged.serialize",
        });
        REQUIRE(bundle.staged->value == "before");
        REQUIRE(bundle.block->state() == Block::State::Created);
    }

    SECTION("staged deserialization failure stops before configuration") {
        bundle.probe->stagedDeserializeResult = Result::ERROR;

        REQUIRE(bundle.block->reconfigure(ConfigWithValue("after")) == Result::ERROR);
        REQUIRE(bundle.probe->events == std::vector<std::string>{
            "block.candidate.deserialize",
            "block.validate:after:before",
            "block.staged.serialize",
            "block.staged.deserialize",
        });
        REQUIRE(bundle.staged->value == "before");
        REQUIRE(bundle.block->state() == Block::State::Created);
    }

    SECTION("configuration failure rolls back the staged configuration") {
        bundle.probe->configureResult = Result::ERROR;

        REQUIRE(bundle.block->reconfigure(ConfigWithValue("after")) == Result::ERROR);
        REQUIRE(bundle.block->state() == Block::State::Created);

        // Expected failure: Block::reconfigure keeps the candidate after configure fails.
        REQUIRE(bundle.staged->value == "before");
    }

    SECTION("interface refresh failure rolls back configuration and interface") {
        bundle.probe->defineResult = Result::ERROR;

        REQUIRE(bundle.block->reconfigure(ConfigWithValue("after")) == Result::ERROR);
        REQUIRE(bundle.block->state() == Block::State::Created);

        // Expected failure: Block::reconfigure does not roll back after interface definition fails.
        CHECK(bundle.staged->value == "before");
        CHECK(bundle.block->interface()->outputs().size() == 1);
    }

    SECTION("stored configuration serialization failure is returned") {
        bundle.staged->serializeResult = Result::ERROR;
        Parser::Map serialized;

        REQUIRE(bundle.block->config(serialized) == Result::ERROR);
        REQUIRE(serialized.empty());
        REQUIRE(bundle.probe->events == std::vector<std::string>{"block.staged.serialize"});
    }
}

TEST_CASE("Block reconfigures nested children in validation and commit phases",
          "[core][lifecycle][reconfigure][module]") {
    auto bundle = MakeBlock();
    ScopedChildRegistration registration(bundle.probe);
    SchedulerHarness scheduler;
    bundle.probe->children = {"first", "second"};
    bundle.probe->recordChildReconfigureEvents = true;

    REQUIRE(registration.result == Result::SUCCESS);
    REQUIRE(scheduler.createResult == Result::SUCCESS);
    REQUIRE(bundle.block->create("lifecycle-block",
                                 DeviceType::CPU,
                                 RuntimeType::NATIVE,
                                 kLifecycleProvider,
                                 ConfigWithValue("before"),
                                 {},
                                 MakeBlockContext(scheduler.scheduler)) == Result::SUCCESS);
    REQUIRE(bundle.impl->childStagedValue("first") == "before");
    REQUIRE(bundle.impl->childStagedValue("second") == "before");
    bundle.probe->events.clear();

    SECTION("all child validations precede every commit") {
        REQUIRE(bundle.block->reconfigure(ConfigWithValue("after")) == Result::SUCCESS);
        REQUIRE(EventsStartingWith(bundle.probe->events, "child.") ==
                std::vector<std::string>{
                    "child.validate:lifecycle-block-first:after:before",
                    "child.validate:lifecycle-block-second:after:before",
                    "child.validate:lifecycle-block-first:after:before",
                    "child.reconfigure:lifecycle-block-first:after:before",
                    "child.validate:lifecycle-block-second:after:before",
                    "child.reconfigure:lifecycle-block-second:after:before",
                });
        REQUIRE(bundle.staged->value == "after");
        REQUIRE(bundle.impl->childConfigValue("first") == "after");
        REQUIRE(bundle.impl->childConfigValue("second") == "after");
        REQUIRE(bundle.impl->childStagedValue("first") == "after");
        REQUIRE(bundle.impl->childStagedValue("second") == "after");
        REQUIRE(bundle.block->state() == Block::State::Created);
    }

    SECTION("validation failure rolls parent sources back before any child commit") {
        bundle.probe->failingChildValidate = "second";

        REQUIRE(bundle.block->reconfigure(ConfigWithValue("after")) == Result::ERROR);
        REQUIRE(EventsStartingWith(bundle.probe->events, "child.") ==
                std::vector<std::string>{
                    "child.validate:lifecycle-block-first:after:before",
                    "child.validate:lifecycle-block-second:after:before",
                });
        REQUIRE(bundle.staged->value == "before");
        REQUIRE(bundle.impl->childConfigValue("first") == "before");
        REQUIRE(bundle.impl->childConfigValue("second") == "before");
        REQUIRE(bundle.impl->childStagedValue("first") == "before");
        REQUIRE(bundle.impl->childStagedValue("second") == "before");
        REQUIRE(bundle.impl->childCandidateValue("first") == "after");
        REQUIRE(bundle.impl->childCandidateValue("second") == "after");
        REQUIRE(bundle.block->state() == Block::State::Created);
    }

    SECTION("partial child commit failure rolls back parent and committed siblings") {
        bundle.probe->failingChildReconfigure = "second";

        REQUIRE(bundle.block->reconfigure(ConfigWithValue("after")) == Result::ERROR);
        REQUIRE(EventsStartingWith(bundle.probe->events, "child.") ==
                std::vector<std::string>{
                    "child.validate:lifecycle-block-first:after:before",
                    "child.validate:lifecycle-block-second:after:before",
                    "child.validate:lifecycle-block-first:after:before",
                    "child.reconfigure:lifecycle-block-first:after:before",
                    "child.validate:lifecycle-block-second:after:before",
                    "child.reconfigure:lifecycle-block-second:after:before",
                });
        REQUIRE(bundle.impl->childStagedValue("second") == "before");
        REQUIRE(bundle.block->state() == Block::State::Created);

        // Expected failure: a child commit error leaves the parent configuration committed.
        CHECK(bundle.staged->value == "before");
        // Expected failure: child source configurations are not rolled back after partial commit.
        CHECK(bundle.impl->childConfigValue("first") == "before");
        // Expected failure: an already committed sibling is not rolled back.
        CHECK(bundle.impl->childStagedValue("first") == "before");
    }

    SECTION("rollback deserialization failure leaves an explicit errored state") {
        bundle.probe->failingChildValidate = "second";
        bundle.probe->stagedDeserializeFailureValue = "before";

        REQUIRE(bundle.block->reconfigure(ConfigWithValue("after")) == Result::ERROR);
        REQUIRE(bundle.impl->childStagedValue("first") == "before");
        REQUIRE(bundle.impl->childStagedValue("second") == "before");

        // Expected failure: failed rollback leaves the newly committed parent configuration.
        CHECK(bundle.staged->value == "before");
        // Expected failure: failed rollback is not reflected in the block lifecycle state.
        CHECK(bundle.block->state() == Block::State::Errored);
    }
}

TEST_CASE("Errored blocks require recreation", "[core][lifecycle][reconfigure]") {
    auto bundle = MakeBlock();
    bundle.probe->createResult = Result::ERROR;

    REQUIRE(bundle.block->create("lifecycle-block",
                                 DeviceType::CPU,
                                 RuntimeType::NATIVE,
                                 kLifecycleProvider,
                                 ConfigWithValue("before"),
                                 {},
                                 MakeBlockContext()) == Result::ERROR);
    REQUIRE(bundle.block->state() == Block::State::Errored);

    bundle.probe->events.clear();
    REQUIRE(bundle.block->reconfigure(ConfigWithValue("after")) == Result::RECREATE);
    REQUIRE(bundle.probe->events == std::vector<std::string>{"block.candidate.deserialize"});
    REQUIRE(bundle.staged->value == "before");
}

TEST_CASE("Module reconfiguration separates validation from commit", "[core][lifecycle][reconfigure]") {
    auto bundle = MakeModule();

    REQUIRE(bundle.module->create("lifecycle-module",
                                  ConfigWithValue("before"),
                                  {}) == Result::SUCCESS);
    bundle.probe->events.clear();
    bundle.probe->identityReady.clear();

    SECTION("unchanged configuration skips validation and the hook") {
        REQUIRE(bundle.module->reconfigure(ConfigWithValue("before")) == Result::SUCCESS);
        REQUIRE(bundle.probe->events == std::vector<std::string>{"module.candidate.deserialize"});
        REQUIRE(bundle.staged->value == "before");
    }

    SECTION("candidate deserialization failure stops before validation") {
        bundle.probe->candidateDeserializeResult = Result::ERROR;

        REQUIRE(bundle.module->reconfigure(ConfigWithValue("after")) == Result::ERROR);
        REQUIRE(bundle.probe->events ==
                std::vector<std::string>{"module.candidate.deserialize"});
        REQUIRE(bundle.staged->value == "before");
        REQUIRE(bundle.probe->identityReady.empty());
    }

    SECTION("validation-only does not invoke or commit reconfiguration") {
        REQUIRE(bundle.module->reconfigure(ConfigWithValue("after"), true) == Result::SUCCESS);
        REQUIRE(bundle.probe->events == std::vector<std::string>{
            "module.candidate.deserialize",
            "module.validate:after:before",
        });
        REQUIRE(bundle.staged->value == "before");
    }

    SECTION("successful hook commits the candidate") {
        REQUIRE(bundle.module->reconfigure(ConfigWithValue("after")) == Result::SUCCESS);
        REQUIRE(bundle.probe->events == std::vector<std::string>{
            "module.candidate.deserialize",
            "module.validate:after:before",
            "module.reconfigure:after:before",
        });
        REQUIRE(bundle.staged->value == "after");
        REQUIRE(bundle.probe->identityReady == std::vector<bool>{true, true});
    }

    SECTION("validation failure preserves the staged configuration") {
        bundle.probe->validateResult = Result::ERROR;

        REQUIRE(bundle.module->reconfigure(ConfigWithValue("after")) == Result::ERROR);
        REQUIRE(bundle.probe->events == std::vector<std::string>{
            "module.candidate.deserialize",
            "module.validate:after:before",
        });
        REQUIRE(bundle.staged->value == "before");
    }

    SECTION("recreate validation result is propagated before the hook") {
        bundle.probe->validateResult = Result::RECREATE;

        REQUIRE(bundle.module->reconfigure(ConfigWithValue("after")) == Result::RECREATE);
        REQUIRE(bundle.probe->events == std::vector<std::string>{
            "module.candidate.deserialize",
            "module.validate:after:before",
        });
        REQUIRE(bundle.staged->value == "before");
    }

    SECTION("hook failure preserves the staged configuration") {
        bundle.probe->reconfigureResult = Result::ERROR;

        REQUIRE(bundle.module->reconfigure(ConfigWithValue("after")) == Result::ERROR);
        REQUIRE(bundle.probe->events.back() == "module.reconfigure:after:before");
        REQUIRE(bundle.staged->value == "before");
    }

    SECTION("partial hook mutation is rolled back on failure") {
        bundle.probe->reconfigureResult = Result::ERROR;
        bundle.probe->commitBeforeReconfigureFailure = true;

        REQUIRE(bundle.module->reconfigure(ConfigWithValue("after")) == Result::ERROR);
        REQUIRE(bundle.probe->events.back() == "module.reconfigure:after:before");

        // Expected failure: Module::reconfigure has no snapshot for partial hook mutation.
        CHECK(bundle.staged->value == "before");
    }

    SECTION("recreate hook result preserves the staged configuration") {
        bundle.probe->reconfigureResult = Result::RECREATE;

        REQUIRE(bundle.module->reconfigure(ConfigWithValue("after")) == Result::RECREATE);
        REQUIRE(bundle.probe->events.back() == "module.reconfigure:after:before");
        REQUIRE(bundle.staged->value == "before");
    }
}
