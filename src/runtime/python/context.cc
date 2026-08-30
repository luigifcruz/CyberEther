#include <jetstream/runtime_context_python.hh>

#include <algorithm>
#include <functional>
#include <mutex>
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
    bool scheduled = false;
};

struct PythonDependencyRegistry {
    std::mutex mutex;
    std::unordered_map<PythonRuntimeContext*, PythonDependencyRecord> records;
    PythonDependencyEnvironment activeEnvironment;
    U64 generation = 0;
};

PythonDependencyRegistry& DependencyRegistry() {
    static PythonDependencyRegistry registry;
    return registry;
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
        ++registry.generation;
    }
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
        ++registry.generation;
    }
    return Result::SUCCESS;
}

PythonDependencySnapshot SnapshotPythonDependencies() {
    std::lock_guard<std::recursive_mutex> operationLock(PythonOperationMutex());
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

    // Resolve the installation policy for the declared PEP 723 metadata. The
    // dependency installer will consume the decision once installation lands.
    (void)ResolvePythonDependencyPolicy(metadata);

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
    return pimpl->run();
}

Result PythonRuntimeContext::computeDeinitialize() {
    return UnschedulePythonDependencies(this);
}

}  // namespace Jetstream
