#ifndef JETSTREAM_RUNTIME_CONTEXT_PYTHON_HH
#define JETSTREAM_RUNTIME_CONTEXT_PYTHON_HH

#include <memory>
#include <string>

#include "jetstream/module_interface.hh"
#include "jetstream/runtime.hh"
#include "jetstream/runtime_context.hh"
#include "jetstream/tensor_link.hh"

namespace Jetstream {

struct PythonRuntimeContext : Runtime::Context {
 public:
    using Diagnostic = Runtime::Context::Diagnostic;

    PythonRuntimeContext();
    ~PythonRuntimeContext();

    Diagnostic diagnostic() const override;

    Result createCompute(const std::string& source,
                         const Module::Interface::EntryList& inputOrder,
                         const TensorMap& inputs,
                         const Module::Interface::EntryList& outputOrder,
                         const TensorMap& outputs);
    Result destroyCompute();

    virtual Result computeInitialize();
    virtual Result computeSubmit();
    virtual Result computeDeinitialize();

 private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;
};

}  // namespace Jetstream

#endif  // JETSTREAM_RUNTIME_CONTEXT_PYTHON_HH
