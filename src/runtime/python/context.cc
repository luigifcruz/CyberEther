#include <jetstream/runtime_context_python.hh>

#include <algorithm>
#include <mutex>
#include <unordered_map>

#include "bridge/base.hh"
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
    std::unordered_map<const PythonRuntimeContext*, PythonDependencyRecord> records;
    U64 generation = 0;
};

PythonDependencyRegistry& DependencyRegistry() {
    static PythonDependencyRegistry registry;
    return registry;
}

}  // namespace

struct PythonRuntimeContext::Impl : Bridge {};

Result StagePythonDependencies(const PythonRuntimeContext* context,
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

Result SchedulePythonDependencies(const PythonRuntimeContext* context) {
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

Result UnschedulePythonDependencies(const PythonRuntimeContext* context) {
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

Result RemovePythonDependencies(const PythonRuntimeContext* context) {
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
    const auto stopResult = pimpl->stop();
    const auto removeResult = RemovePythonDependencies(this);
    return stopResult == Result::SUCCESS ? removeResult : stopResult;
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
