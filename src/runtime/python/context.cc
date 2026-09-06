#include <jetstream/runtime_context_python.hh>

#include "bridge/base.hh"
#include "runtime/helpers.hh"
#include "runtime/python/dependencies/coordinator.hh"
#include "runtime/python/dependencies/base.hh"

namespace Jetstream {

struct PythonRuntimeContext::Impl : Bridge {
    PythonDependencyMetadata validatedMetadata;
};

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

    std::lock_guard<std::recursive_mutex> operationLock(PythonOperationMutex());
    // The selected interpreter is fixed until restart. Revalidate when metadata
    // changes; environment reloads can reuse this context's successful validation.
    if (metadata != pimpl->validatedMetadata) {
        JST_CHECK(ValidatePythonDependencyMetadata(metadata));
        pimpl->validatedMetadata = metadata;
    }

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
    pimpl->validatedMetadata = {};
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
