#ifndef JETSTREAM_RUNTIME_PYTHON_BRIDGE_BASE_HH
#define JETSTREAM_RUNTIME_PYTHON_BRIDGE_BASE_HH

#include <mutex>
#include <string>
#include <vector>

#include "jetstream/module_interface.hh"
#include "jetstream/runtime.hh"
#include "jetstream/runtime_context.hh"
#include "jetstream/tensor_link.hh"
#include "runtime/python/bridge/cpython/base.hh"

namespace Jetstream {

struct Bridge {
 public:
    Result start(const std::string& source,
                 const Module::Interface::EntryList& inputOrder,
                 const TensorMap& inputs,
                 const Module::Interface::EntryList& outputOrder,
                 const TensorMap& outputs);
    Result stop();
    Result run();
    Runtime::Context::Diagnostic diagnostic() const;

 protected:
    Bridge() = default;
    ~Bridge();

    Bridge(const Bridge&) = delete;
    Bridge& operator=(const Bridge&) = delete;

 private:
    CPython::PyObject* globals = nullptr;
    CPython::PyObject* runner = nullptr;

    //
    // Status (Health) [bridge/status.cc]
    //

    bool healthy = true;
    std::string status;
    mutable std::mutex statusMutex;

    void setInfo(const std::string& text);
    void setError(const std::string& text, std::string details = "");

    //
    // Console Logs (Poll) [bridge/console.cc]
    //

    std::vector<std::string> consoleLines;
    mutable std::mutex consoleMutex;

    void consoleClear();
    void consoleAppend(const std::string& text);
    bool consoleRefresh();

    //
    // Tensor IO (with Attributes) [bridge/tensor.cc]
    //

    CPython::PyObject* createTensorContext(const Module::Interface::EntryList& inputOrder,
                                           const TensorMap& inputs,
                                           const Module::Interface::EntryList& outputOrder,
                                           const TensorMap& outputs);
};

}  // namespace Jetstream

#endif  // JETSTREAM_RUNTIME_PYTHON_BRIDGE_BASE_HH
