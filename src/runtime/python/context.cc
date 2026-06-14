#include <jetstream/runtime_context_python.hh>

#include "bridge/base.hh"

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
                                           const Module::Interface::EntryList& inputOrder,
                                           const TensorMap& inputs,
                                           const Module::Interface::EntryList& outputOrder,
                                           const TensorMap& outputs) {
    return pimpl->start(source, inputOrder, inputs, outputOrder, outputs);
}

Result PythonRuntimeContext::destroyCompute() {
    return pimpl->stop();
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
