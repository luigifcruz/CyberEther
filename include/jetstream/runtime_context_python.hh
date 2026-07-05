#ifndef JETSTREAM_RUNTIME_CONTEXT_PYTHON_HH
#define JETSTREAM_RUNTIME_CONTEXT_PYTHON_HH

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "jetstream/flowgraph.hh"
#include "jetstream/module_interface.hh"
#include "jetstream/runtime.hh"
#include "jetstream/runtime_context.hh"
#include "jetstream/tensor_link.hh"

namespace Jetstream {

struct JETSTREAM_API PythonRuntimeContext : Runtime::Context {
 public:
    using Diagnostic = Runtime::Context::Diagnostic;

    struct Candidate {
        std::string label;
        std::string path;
        std::string libraryPath;
    };

    struct Validation {
        bool valid = false;
        std::string inputPath;
        std::string libraryPath;
        std::string programPath;
        std::string message;
        std::vector<std::string> attempts;
    };

    PythonRuntimeContext();
    ~PythonRuntimeContext();

    Diagnostic diagnostic() const override;

    Result createCompute(const std::string& source,
                         const std::unordered_map<std::string, std::string>& pieces,
                         const Module::Interface::EntryList& inputOrder,
                         const TensorMap& inputs,
                         const Module::Interface::EntryList& outputOrder,
                         const TensorMap& outputs,
                         const std::shared_ptr<Flowgraph::Environment>& environment = nullptr,
                         const std::shared_ptr<Flowgraph::View>& view = nullptr);
    Result destroyCompute();

    virtual Result computeInitialize();
    virtual Result computeSubmit();
    virtual Result computeDeinitialize();

    static Validation ValidateRuntimePath(const std::string& path);
    static std::vector<Candidate> DiscoverRuntimes();

 private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;
};

}  // namespace Jetstream

#endif  // JETSTREAM_RUNTIME_CONTEXT_PYTHON_HH
