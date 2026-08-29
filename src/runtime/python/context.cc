#include <jetstream/runtime_context_python.hh>

#include "bridge/base.hh"
#include "runtime/helpers.hh"
#include "runtime/python/dependencies/base.hh"

namespace Jetstream {

struct PythonRuntimeContext::Impl : Bridge {};

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

    return pimpl->start(expandedSource, inputOrder, inputs, outputOrder, outputs, environment, view);
}

Result PythonRuntimeContext::destroyCompute() {
    return pimpl->stop();
}

void PythonRuntimeContext::setImmutableOutputAttributes(
    const std::vector<std::unordered_set<std::string>>& keys) {
    pimpl->setImmutableOutputAttributes(keys);
}

Result PythonRuntimeContext::computeInitialize() {
    return Result::SUCCESS;
}

Result PythonRuntimeContext::computeSubmit() {
    return pimpl->run();
}

Result PythonRuntimeContext::computeDeinitialize() {
    return Result::SUCCESS;
}

}  // namespace Jetstream
