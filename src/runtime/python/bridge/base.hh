#ifndef JETSTREAM_RUNTIME_PYTHON_BRIDGE_BASE_HH
#define JETSTREAM_RUNTIME_PYTHON_BRIDGE_BASE_HH

#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "jetstream/module_interface.hh"
#include "jetstream/runtime.hh"
#include "jetstream/runtime_context.hh"
#include "jetstream/runtime_context_python.hh"
#include "jetstream/tensor_link.hh"
#include "runtime/python/bridge/cpython/base.hh"

namespace Jetstream {

struct Bridge {
 public:
    Result start(const std::string& source,
                 const Module::Interface::EntryList& inputOrder,
                 const TensorMap& inputs,
                 const Module::Interface::EntryList& outputOrder,
                 const TensorMap& outputs,
                 const std::shared_ptr<Flowgraph::Environment>& environment = nullptr,
                 const std::shared_ptr<Flowgraph::View>& view = nullptr);
    Result stop();
    Result run();
    Runtime::Context::Diagnostic diagnostic() const;

 protected:
    Bridge() = default;
    ~Bridge();

    Bridge(const Bridge&) = delete;
    Bridge& operator=(const Bridge&) = delete;

 private:
    mutable std::recursive_mutex lifecycleMutex;

    CPython::PyObject* globals = nullptr;
    CPython::PyObject* runner = nullptr;

    CPython::PyObject* valueConverters = nullptr;
    bool valueConvertersFetched = false;

    CPython::PyObject* valueConverterTable();

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
    std::vector<std::string> consoleTailLines;
    mutable std::mutex consoleMutex;

    void consoleClear();
    void consoleAppend(const std::string& text);
    bool consoleRefresh();

    //
    // Tensor IO (with Attributes) [bridge/tensor.cc]
    //

    struct AttributePort {
        Tensor tensor;
        CPython::PyObject* dict = nullptr;
        std::unordered_map<std::string, CPython::PyObject*> snapshot;
    };

    std::vector<AttributePort> inputAttributePorts;
    std::vector<AttributePort> outputAttributePorts;

    CPython::PyObject* createTensorContext(const Module::Interface::EntryList& inputOrder,
                                           const TensorMap& inputs,
                                           const Module::Interface::EntryList& outputOrder,
                                           const TensorMap& outputs);
    CPython::PyObject* createAttributeDicts(const Module::Interface::EntryList& order,
                                            const TensorMap& tensors,
                                            std::vector<AttributePort>& ports);
    void refreshAttributes();
    void flushAttributes();
    void destroyAttributePorts();

    //
    // Environment IO [bridge/environment.cc]
    //

    std::shared_ptr<Flowgraph::Environment> environment;
    CPython::PyObject* environmentDict = nullptr;
    U64 environmentEpoch = 0;
    bool environmentSynced = false;

    CPython::PyObject* createEnvironmentDict();
    void refreshEnvironment();
    void flushEnvironment();
    void trackEnvironment();
    void destroyEnvironmentDict();

    //
    // Metrics IO [bridge/metrics.cc]
    //

    std::shared_ptr<Flowgraph::View> flowgraphView;
    CPython::PyObject* metricsDict = nullptr;
    std::unordered_set<std::string> metricsRequests;
    bool metricsSubscribeAll = false;

    CPython::PyObject* createMetricsDict();
    void refreshMetrics();
    void collectMetricsRequests();
    void destroyMetricsDict();
};

}  // namespace Jetstream

#endif  // JETSTREAM_RUNTIME_PYTHON_BRIDGE_BASE_HH
