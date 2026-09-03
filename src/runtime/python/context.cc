#include <jetstream/runtime_context_python.hh>

#include <algorithm>
#include <functional>
#include <limits>
#include <mutex>
#include <optional>
#include <string_view>
#include <tuple>
#include <unordered_map>

#include "bridge/base.hh"
#include "runtime/python/bridge/bootstrap/base.hh"
#include "runtime/helpers.hh"
#include "runtime/python/context.hh"
#include "runtime/python/dependencies/base.hh"

namespace Jetstream {

namespace {

struct PythonDependencyRecord {
    std::vector<std::string> requirements;
    std::string block;
    std::weak_ptr<Flowgraph::View> view;
    bool scheduled = false;
};

struct PythonDependencyRegistry {
    std::mutex mutex;
    std::unordered_map<PythonRuntimeContext*, PythonDependencyRecord> records;
    PythonDependencyEnvironment activeEnvironment;
    PythonDependencyRequest request;
    std::vector<std::string> resolvedRequirements;
    std::optional<PythonDependencyPolicy> dependencyPolicy;
    U64 resolvedGeneration = std::numeric_limits<U64>::max();
    Result resolvedResult = Result::INCOMPLETE;
    U64 generation = 0;
};

PythonDependencyRegistry& DependencyRegistry() {
    static PythonDependencyRegistry registry;
    return registry;
}

void InvalidateDependencyResolution(PythonDependencyRegistry& registry) {
    registry.resolvedGeneration = std::numeric_limits<U64>::max();
    if (registry.request.state != PythonDependencyRequestState::Installing) {
        registry.request = {};
    }
}

}  // namespace

struct PythonRuntimeContext::Impl : Bridge {};

Result StagePythonDependencies(PythonRuntimeContext* context,
                               const std::vector<std::string>& requirements) {
    if (!context) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Cannot stage dependencies for a null "
                  "context.");
        return Result::ERROR;
    }

    std::lock_guard<std::recursive_mutex> operationLock(PythonOperationMutex());
    auto& registry = DependencyRegistry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    auto [entry, inserted] = registry.records.try_emplace(context);
    if (inserted || entry->second.requirements != requirements) {
        entry->second.requirements = requirements;
        InvalidateDependencyResolution(registry);
        ++registry.generation;
    }
    return Result::SUCCESS;
}

Result SetPythonDependencyPolicy(const std::string& value) {
    PythonDependencyPolicy policy;
    JST_CHECK(ParsePythonDependencyPolicy(value, policy));

    auto& registry = DependencyRegistry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    if (registry.dependencyPolicy == policy) {
        return Result::SUCCESS;
    }

    registry.dependencyPolicy = policy;
    InvalidateDependencyResolution(registry);
    ++registry.generation;
    return Result::SUCCESS;
}

Result SetPythonDependencyOrigin(PythonRuntimeContext* context,
                                 const std::string& block,
                                 const std::shared_ptr<Flowgraph::View>& view) {
    if (!context) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Cannot set the origin for a null "
                  "dependency context.");
        return Result::ERROR;
    }

    auto& registry = DependencyRegistry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    auto& record = registry.records[context];
    record.block = block;
    record.view = view;
    return Result::SUCCESS;
}

Result SchedulePythonDependencies(PythonRuntimeContext* context) {
    if (!context) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Cannot schedule a null dependency "
                  "context.");
        return Result::ERROR;
    }

    std::lock_guard<std::recursive_mutex> operationLock(PythonOperationMutex());
    auto& registry = DependencyRegistry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    auto& record = registry.records[context];
    if (!record.scheduled) {
        record.scheduled = true;
        InvalidateDependencyResolution(registry);
        ++registry.generation;
    }
    return Result::SUCCESS;
}

Result UnschedulePythonDependencies(PythonRuntimeContext* context) {
    if (!context) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Cannot unschedule a null dependency "
                  "context.");
        return Result::ERROR;
    }

    std::lock_guard<std::recursive_mutex> operationLock(PythonOperationMutex());
    auto& registry = DependencyRegistry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    const auto entry = registry.records.find(context);
    if (entry != registry.records.end() && entry->second.scheduled) {
        entry->second.scheduled = false;
        InvalidateDependencyResolution(registry);
        ++registry.generation;
    }
    return Result::SUCCESS;
}

Result RemovePythonDependencies(PythonRuntimeContext* context) {
    if (!context) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Cannot remove a null dependency context.");
        return Result::ERROR;
    }

    std::lock_guard<std::recursive_mutex> operationLock(PythonOperationMutex());
    auto& registry = DependencyRegistry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    if (registry.records.erase(context) > 0) {
        InvalidateDependencyResolution(registry);
        ++registry.generation;
    }
    return Result::SUCCESS;
}

PythonDependencySnapshot SnapshotPythonDependencies() {
    auto& registry = DependencyRegistry();
    std::lock_guard<std::mutex> lock(registry.mutex);

    PythonDependencySnapshot snapshot = {
        .generation = registry.generation,
    };
    for (const auto& [_, record] : registry.records) {
        if (!record.scheduled) {
            continue;
        }
        snapshot.requirements.insert(snapshot.requirements.end(),
                                     record.requirements.begin(),
                                     record.requirements.end());
    }
    std::ranges::sort(snapshot.requirements);
    snapshot.requirements.erase(
        std::unique(snapshot.requirements.begin(), snapshot.requirements.end()),
        snapshot.requirements.end());
    return snapshot;
}

PythonDependencyStateSnapshot SnapshotPythonDependencyState() {
    auto& registry = DependencyRegistry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    return {
        .generation = registry.generation,
        .request = registry.request,
    };
}

PythonDependencyRequest GetPythonDependencyRequest() {
    auto& registry = DependencyRegistry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    return registry.request;
}

Result ReconcilePythonDependencies() {
    auto& registry = DependencyRegistry();
    PythonDependencySnapshot snapshot;
    std::vector<PythonDependencyRequestEntry> entries;
    std::optional<PythonDependencyPolicy> dependencyPolicy;
    {
        std::lock_guard<std::mutex> lock(registry.mutex);
        if (registry.request.state == PythonDependencyRequestState::Installing) {
            return Result::INCOMPLETE;
        }
        if (registry.resolvedGeneration == registry.generation) {
            return registry.resolvedResult;
        }

        snapshot.generation = registry.generation;
        dependencyPolicy = registry.dependencyPolicy;
        for (const auto& [_, record] : registry.records) {
            if (!record.scheduled) {
                continue;
            }
            snapshot.requirements.insert(snapshot.requirements.end(),
                                         record.requirements.begin(),
                                         record.requirements.end());
            for (const auto& requirement : record.requirements) {
                entries.push_back({requirement, record.block, record.view});
            }
        }
    }
    std::ranges::sort(snapshot.requirements);
    snapshot.requirements.erase(
        std::unique(snapshot.requirements.begin(), snapshot.requirements.end()),
        snapshot.requirements.end());
    std::ranges::sort(entries, [](const auto& lhs, const auto& rhs) {
        return std::tie(lhs.requirement, lhs.block) <
               std::tie(rhs.requirement, rhs.block);
    });

    if (snapshot.requirements.empty()) {
        std::lock_guard<std::mutex> lock(registry.mutex);
        if (registry.resolvedRequirements.empty()) {
            registry.request = {};
            registry.resolvedGeneration = snapshot.generation;
            registry.resolvedResult = Result::SUCCESS;
            return Result::SUCCESS;
        }
    }

    PythonDependencyEnvironment environment;
    auto result = PreparePythonDependencyEnvironment(
        snapshot.requirements, false, environment);
    if (result == Result::INCOMPLETE) {
        const PythonDependencyMetadata metadata = {
            .requirements = snapshot.requirements,
        };
        const auto decision = dependencyPolicy.has_value()
                                  ? ResolvePythonDependencyPolicy(metadata,
                                                                  dependencyPolicy.value())
                                  : ResolvePythonDependencyPolicy(metadata);
        if (decision.policy == PythonDependencyPolicy::Deny) {
            std::lock_guard<std::mutex> lock(registry.mutex);
            if (registry.generation != snapshot.generation) {
                return Result::INCOMPLETE;
            }
            registry.request = {
                .generation = snapshot.generation,
                .state = PythonDependencyRequestState::Denied,
                .dependencies = std::move(entries),
                .message = "Dependency installation is disabled by policy.",
            };
            registry.resolvedGeneration = snapshot.generation;
            registry.resolvedResult = Result::INCOMPLETE;
            return Result::INCOMPLETE;
        }
        if (decision.consentRequired) {
            std::lock_guard<std::mutex> lock(registry.mutex);
            if (registry.generation != snapshot.generation) {
                return Result::INCOMPLETE;
            }
            registry.request = {
                .generation = snapshot.generation,
                .state = PythonDependencyRequestState::ApprovalRequired,
                .dependencies = std::move(entries),
            };
            registry.resolvedGeneration = snapshot.generation;
            registry.resolvedResult = Result::INCOMPLETE;
            return Result::INCOMPLETE;
        }
        if (decision.installAllowed) {
            result = PreparePythonDependencyEnvironment(
                snapshot.requirements, true, environment);
        }
    }

    if (result == Result::SUCCESS) {
        {
            std::lock_guard<std::mutex> lock(registry.mutex);
            if (registry.generation != snapshot.generation) {
                return Result::INCOMPLETE;
            }
        }
        result = ActivatePythonDependencyEnvironment(environment);
    }

    std::lock_guard<std::mutex> lock(registry.mutex);
    if (registry.generation != snapshot.generation) {
        return Result::INCOMPLETE;
    }
    registry.resolvedGeneration = snapshot.generation;
    registry.resolvedResult = result == Result::SUCCESS ? Result::SUCCESS
                                                        : Result::INCOMPLETE;
    if (result == Result::SUCCESS) {
        registry.resolvedRequirements = snapshot.requirements;
        registry.request = {};
    } else if (result == Result::INCOMPLETE) {
        registry.request = {};
    } else {
        registry.request = {
            .generation = snapshot.generation,
            .state = PythonDependencyRequestState::Failed,
            .dependencies = std::move(entries),
            .message = JST_LOG_LAST_ERROR(),
        };
    }
    return registry.resolvedResult;
}

Result BeginPythonDependencyInstallation(U64 generation) {
    auto& registry = DependencyRegistry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    if (registry.generation != generation ||
        registry.request.generation != generation ||
        (registry.request.state != PythonDependencyRequestState::ApprovalRequired &&
         registry.request.state != PythonDependencyRequestState::Failed)) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] The dependency approval request is stale.");
        return Result::ERROR;
    }

    PythonDependencyMetadata metadata;
    metadata.requirements.reserve(registry.request.dependencies.size());
    for (const auto& dependency : registry.request.dependencies) {
        metadata.requirements.push_back(dependency.requirement);
    }
    const auto decision = registry.dependencyPolicy.has_value()
                              ? ResolvePythonDependencyPolicy(
                                    metadata, registry.dependencyPolicy.value())
                              : ResolvePythonDependencyPolicy(metadata);
    if (decision.policy == PythonDependencyPolicy::Deny) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Dependency installation is disabled "
                  "by policy.");
        return Result::ERROR;
    }

    registry.request.state = PythonDependencyRequestState::Installing;
    registry.request.output.clear();
    registry.request.message.clear();
    registry.request.approved = true;
    return Result::SUCCESS;
}

Result InstallPythonDependencies(U64 generation) {
    auto& registry = DependencyRegistry();
    std::vector<std::string> requirements;
    {
        std::lock_guard<std::mutex> lock(registry.mutex);
        if (registry.generation != generation ||
            registry.request.generation != generation ||
            registry.request.state != PythonDependencyRequestState::Installing) {
            if (registry.request.generation == generation &&
                registry.request.state == PythonDependencyRequestState::Installing) {
                registry.request.state = PythonDependencyRequestState::Failed;
                registry.request.message = "The dependency list changed before installation started.";
            }
            JST_ERROR("[RUNTIME_CONTEXT_PYTHON] The dependency installation request "
                      "is stale.");
            return Result::ERROR;
        }
        for (const auto& dependency : registry.request.dependencies) {
            requirements.push_back(dependency.requirement);
        }
    }
    std::ranges::sort(requirements);
    requirements.erase(std::unique(requirements.begin(), requirements.end()),
                       requirements.end());

    PythonDependencyEnvironment environment;
    const auto onOutput = [&registry, generation](std::string_view output) {
        std::lock_guard<std::mutex> lock(registry.mutex);
        if (registry.request.generation != generation ||
            registry.request.state != PythonDependencyRequestState::Installing) {
            return;
        }
        registry.request.output.append(output.data(), output.size());
    };
    const auto prepareResult = PreparePythonDependencyEnvironment(
        requirements, true, environment, onOutput);
    if (prepareResult == Result::SUCCESS) {
        std::lock_guard<std::mutex> lock(registry.mutex);
        if (registry.generation != generation) {
            registry.request.state = PythonDependencyRequestState::Failed;
            registry.request.message = "The dependency list changed.";
            registry.resolvedGeneration = generation;
            registry.resolvedResult = Result::INCOMPLETE;
            JST_ERROR("[RUNTIME_CONTEXT_PYTHON] {}", registry.request.message);
            return Result::ERROR;
        }
    }

    auto result = prepareResult;
    if (result == Result::SUCCESS) {
        result = ActivatePythonDependencyEnvironment(environment);
    }

    std::lock_guard<std::mutex> lock(registry.mutex);
    if (result == Result::SUCCESS && registry.generation == generation) {
        registry.request.state = PythonDependencyRequestState::Installed;
        registry.request.message.clear();
        if (!registry.request.output.empty() &&
            !registry.request.output.ends_with('\n')) {
            registry.request.output.push_back('\n');
        }
        registry.request.output += "Installation completed.\n";
        registry.resolvedRequirements = requirements;
        registry.resolvedGeneration = generation;
        registry.resolvedResult = Result::SUCCESS;
        return Result::SUCCESS;
    }

    registry.request.state = PythonDependencyRequestState::Failed;
    registry.request.message = JST_LOG_LAST_ERROR();
    registry.resolvedGeneration = generation;
    registry.resolvedResult = Result::INCOMPLETE;
    return result == Result::SUCCESS ? Result::ERROR : result;
}

Result ActivatePythonDependencyEnvironment(const PythonDependencyEnvironment& environment) {
    std::lock_guard<std::recursive_mutex> operationLock(PythonOperationMutex());
    auto& registry = DependencyRegistry();

    std::vector<PythonRuntimeContext*> contexts;
    PythonDependencyEnvironment previous;
    {
        std::lock_guard<std::mutex> lock(registry.mutex);
        if (registry.activeEnvironment.key == environment.key &&
            registry.activeEnvironment.sitePackagesPath ==
                environment.sitePackagesPath) {
            return Result::SUCCESS;
        }

        previous = registry.activeEnvironment;
        for (const auto& [context, record] : registry.records) {
            if (record.scheduled) {
                contexts.push_back(context);
            }
        }
    }
    std::ranges::sort(contexts, std::less<PythonRuntimeContext*>());

    std::vector<PythonRuntimeContext*> unloaded;
    for (auto* context : contexts) {
        const auto result = context->unloadCompute();
        if (result != Result::SUCCESS) {
            for (auto* previousContext : unloaded) {
                (void)previousContext->loadCompute();
            }
            return result;
        }
        unloaded.push_back(context);
    }

    const auto switchResult = SwitchPythonEnvironmentPath(
        previous.sitePackagesPath, environment.sitePackagesPath);
    if (switchResult != Result::SUCCESS) {
        (void)SwitchPythonEnvironmentPath(environment.sitePackagesPath,
                                          previous.sitePackagesPath);
        for (auto* context : contexts) {
            (void)context->loadCompute();
        }
        return switchResult;
    }

    for (auto* context : contexts) {
        const auto result = context->loadCompute();
        if (result == Result::SUCCESS) {
            continue;
        }

        for (auto* loadedContext : contexts) {
            (void)loadedContext->unloadCompute();
        }
        (void)SwitchPythonEnvironmentPath(environment.sitePackagesPath,
                                          previous.sitePackagesPath);
        for (auto* previousContext : contexts) {
            (void)previousContext->loadCompute();
        }
        return result;
    }

    {
        std::lock_guard<std::mutex> lock(registry.mutex);
        registry.activeEnvironment = environment;
    }
    return Result::SUCCESS;
}

PythonRuntimeContext::PythonRuntimeContext() {
    pimpl = std::make_unique<Impl>();
}

PythonRuntimeContext::~PythonRuntimeContext() {
    if (pimpl) {
        (void)destroyCompute();
    }
}

PythonRuntimeContext::Diagnostic PythonRuntimeContext::diagnostic() const {
    return pimpl->diagnostic();
}

Result PythonRuntimeContext::createCompute(const std::string& source,
                                           const std::unordered_map<std::string, std::string>& pieces,
                                           const Module::Interface::EntryList& inputOrder,
                                           const TensorMap& inputs,
                                           const Module::Interface::EntryList& outputOrder,
                                           const TensorMap& outputs,
                                           const std::shared_ptr<Flowgraph::Environment>& environment,
                                           const std::shared_ptr<Flowgraph::View>& view) {
    std::string expandedSource = source;
    if (!pieces.empty()) {
        JST_CHECK(ExpandSourcePieces(source, pieces, expandedSource));
        JST_TRACE("[RUNTIME_CONTEXT_PYTHON] Expanded Python source:\n{}", expandedSource);
    }

    PythonDependencyMetadata metadata;
    JST_CHECK(ParsePythonDependencyMetadata(expandedSource, metadata));

    JST_CHECK(ValidatePythonDependencyMetadata(metadata));

    std::lock_guard<std::recursive_mutex> operationLock(PythonOperationMutex());
    const auto startResult = pimpl->start(expandedSource, inputOrder, inputs,
                                          outputOrder, outputs, environment, view);
    if (startResult == Result::SUCCESS ||
        pimpl->diagnostic().status == "Source error.") {
        JST_CHECK(StagePythonDependencies(this, metadata.requirements));
    }
    return startResult;
}

Result PythonRuntimeContext::destroyCompute() {
    std::lock_guard<std::recursive_mutex> operationLock(PythonOperationMutex());
    const auto stopResult = unloadCompute();
    const auto removeResult = RemovePythonDependencies(this);
    return stopResult == Result::SUCCESS ? removeResult : stopResult;
}

Result PythonRuntimeContext::loadCompute() {
    JST_ERROR("[RUNTIME_CONTEXT_PYTHON] The Python module does not provide a "
              "compute loader.");
    return Result::ERROR;
}

Result PythonRuntimeContext::unloadCompute() {
    std::lock_guard<std::recursive_mutex> operationLock(PythonOperationMutex());
    return pimpl->stop();
}

void PythonRuntimeContext::setImmutableOutputAttributes(
    const std::vector<std::unordered_set<std::string>>& keys) {
    pimpl->setImmutableOutputAttributes(keys);
}

Result PythonRuntimeContext::computeInitialize() {
    return SchedulePythonDependencies(this);
}

Result PythonRuntimeContext::computeSubmit() {
    const auto dependencyResult = ReconcilePythonDependencies();
    if (dependencyResult == Result::INCOMPLETE) {
        return Result::SKIP;
    }
    JST_CHECK(dependencyResult);
    return pimpl->run();
}

Result PythonRuntimeContext::computeDeinitialize() {
    return UnschedulePythonDependencies(this);
}

}  // namespace Jetstream
