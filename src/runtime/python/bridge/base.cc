#include "runtime/python/bridge/base.hh"
#include "runtime/python/bridge/cpython/base.hh"
#include "runtime/python/bridge/prelude/bridge.hh"

#include "jetstream/logger.hh"

namespace Jetstream {

using namespace CPython;

namespace {

constexpr const char* kComputeErrorStatus = "Compute error.";

struct GilScope {
    explicit GilScope() : state(PyGILState_Ensure()) {}

    ~GilScope() {
        PyGILState_Release(state);
    }

    GilScope(const GilScope&) = delete;
    GilScope& operator=(const GilScope&) = delete;

    int state;
};

}  // namespace

Bridge::~Bridge() {
    (void)stop();
}

Runtime::Context::Diagnostic Bridge::diagnostic() const {
    Runtime::Context::Diagnostic current;

    {
        std::lock_guard<std::mutex> lock(statusMutex);
        current.healthy = healthy;
        current.status = status;
    }

    {
        std::lock_guard<std::mutex> lock(consoleMutex);
        current.console = consoleLines;
    }

    return current;
}

Result Bridge::start(const std::string& source,
                     const Module::Interface::EntryList& inputOrder,
                     const TensorMap& inputs,
                     const Module::Interface::EntryList& outputOrder,
                     const TensorMap& outputs) {
    const auto loadResult = Py_Load();
    if (loadResult != Result::SUCCESS) {
        setError("Runtime unavailable.");
        return loadResult;
    }

    consoleClear();
    JST_CHECK(stop());

    GilScope gil;

    globals = PyDict_New();
    if (!globals) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Can't create Python globals dictionary.");
        setError("Initialization error.");
        return Result::ERROR;
    }

    auto* bridgeResult = PyRun_StringFlags(kPythonBridge, 257, globals, globals, nullptr);
    if (!bridgeResult) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Can't initialize Python runtime helpers.");
        setError("Initialization error.");
        JST_CHECK(stop());
        return Result::ERROR;
    }
    Py_DecRef(bridgeResult);

    auto getBridgeCallable = [&](const char* name) {
        auto* callable = PyDict_GetItemString(globals, name);
        if (!callable || !PyCallable_Check(callable)) {
            JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Python runtime helper '{}' is unavailable.", name);
            setError("Initialization error.");
            return static_cast<PyObject*>(nullptr);
        }

        return callable;
    };

    auto* loadCompute = getBridgeCallable("_jetstream_load_compute");
    if (!loadCompute) {
        JST_CHECK(stop());
        return Result::ERROR;
    }

    auto* sourceObject = PyUnicode_FromString(source.c_str());
    if (!sourceObject) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Can't prepare Python source execution.");
        setError("Initialization error.");
        JST_CHECK(stop());
        return Result::ERROR;
    }

    auto* compute = PyObject_CallFunctionObjArgs(loadCompute, sourceObject);
    Py_DecRef(sourceObject);
    if (!compute) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Can't compile Python source.");
        setError("Source error.");
        JST_CHECK(stop());
        return Result::ERROR;
    }
    (void)consoleRefresh();

    auto* tensorContext = createTensorContext(inputOrder, inputs, outputOrder, outputs);
    if (!tensorContext) {
        Py_DecRef(compute);
        setError("Tensor conversion error.");
        JST_CHECK(stop());
        return Result::ERROR;
    }

    auto* bindCompute = getBridgeCallable("_jetstream_bind_compute");
    if (!bindCompute) {
        Py_DecRef(tensorContext);
        Py_DecRef(compute);
        JST_CHECK(stop());
        return Result::ERROR;
    }

    auto* boundRunner = PyObject_CallFunctionObjArgs(bindCompute, compute, tensorContext);
    Py_DecRef(tensorContext);
    Py_DecRef(compute);
    if (!boundRunner || !PyCallable_Check(boundRunner)) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Can't create callable Python compute runner.");
        if (boundRunner) { Py_DecRef(boundRunner); }
        setError("Initialization error.");
        JST_CHECK(stop());
        return Result::ERROR;
    }

    runner = boundRunner;

    setInfo("Ready.");

    return Result::SUCCESS;
}

Result Bridge::stop() {
    if (!Py_IsLoaded()) {
        globals = nullptr;
        runner = nullptr;
        return Result::SUCCESS;
    }

    GilScope gil;

    if (runner) {
        Py_DecRef(runner);
        runner = nullptr;
    }

    if (globals) {
        Py_DecRef(globals);
        globals = nullptr;
    }

    return Result::SUCCESS;
}

Result Bridge::run() {
    {
        std::lock_guard<std::mutex> lock(statusMutex);
        if (!healthy && status == kComputeErrorStatus) {
            return Result::SKIP;
        }
    }

    if (!runner || !globals) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Python compute entry has not been created.");
        const auto current = diagnostic();
        if (current.healthy || current.console.empty()) {
            setError("Inactive.", "No valid compute() function is currently loaded.");
        }
        return Result::SKIP;
    }

    const auto loadResult = Py_Load();
    if (loadResult != Result::SUCCESS) {
        setError("Runtime unavailable.");
        return loadResult;
    }

    GilScope gil;

    auto* result = PyObject_CallFunctionObjArgs(runner);
    if (!result) {
        bool alreadyReportingComputeError = false;
        {
            std::lock_guard<std::mutex> lock(statusMutex);
            alreadyReportingComputeError = !healthy && status == kComputeErrorStatus;
        }

        if (!alreadyReportingComputeError) {
            JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Python compute() failed.");
        }
        setError(kComputeErrorStatus);
        return Result::SKIP;
    }
    (void)consoleRefresh();
    Py_DecRef(result);
    setInfo("Running.");

    return Result::SUCCESS;
}

}  // namespace Jetstream
