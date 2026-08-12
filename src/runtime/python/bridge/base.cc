#include "runtime/python/bridge/base.hh"
#include "runtime/python/bridge/convert.hh"
#include "runtime/python/bridge/cpython/base.hh"
#include "runtime/python/bridge/prelude/bridge.hh"

#include <cstddef>

#include "jetstream/logger.hh"

namespace Jetstream {

using namespace CPython;

namespace {

constexpr const char* kComputeErrorStatus = "Compute error.";

struct ThreadStateScope {
    CPython::PyThreadState* state = nullptr;
    std::size_t depth = 0;
};

thread_local ThreadStateScope s_threadState;

std::mutex s_activeBridgeMutex;
std::size_t s_activeBridgeCount = 0;

void RegisterPythonGlobals() {
    std::lock_guard<std::mutex> lock(s_activeBridgeMutex);
    ++s_activeBridgeCount;
}

bool UnregisterPythonGlobals() {
    std::lock_guard<std::mutex> lock(s_activeBridgeMutex);
    if (s_activeBridgeCount == 0) {
        return true;
    }

    --s_activeBridgeCount;
    return s_activeBridgeCount == 0;
}

struct GilScope {
    explicit GilScope() {
        if (s_threadState.depth == 0) {
            if (!s_threadState.state) {
                // Some extension modules cache CPython thread states; deleting
                // and recreating them between computes can leave stale TLS behind.
                s_threadState.state = PyThreadState_New();
                if (!s_threadState.state) {
                    JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Can't create Python thread state.");
                    status = Result::ERROR;
                    return;
                }
            }

            PyEval_RestoreThread(s_threadState.state);
            ownsGil = true;
        }

        ++s_threadState.depth;
        active = true;
    }

    ~GilScope() {
        if (!active) {
            return;
        }

        --s_threadState.depth;
        if (ownsGil && s_threadState.depth == 0) {
            s_threadState.state = PyEval_SaveThread();
        }
    }

    GilScope(const GilScope&) = delete;
    GilScope& operator=(const GilScope&) = delete;

    Result result() const {
        return status;
    }

    bool active = false;
    bool ownsGil = false;
    Result status = Result::SUCCESS;
};

void CallPythonShutdown(PyObject* globals, const char* name) {
    if (!globals) {
        return;
    }

    auto* shutdown = PyDict_GetItemString(globals, name);
    if (!shutdown || !PyCallable_Check(shutdown)) {
        return;
    }

    auto* result = PyObject_CallFunctionObjArgs(shutdown);
    if (!result) {
        PyObject* type = nullptr;
        PyObject* value = nullptr;
        PyObject* traceback = nullptr;
        PyErr_Fetch(&type, &value, &traceback);

        if (type) { Py_DecRef(type); }
        if (value) { Py_DecRef(value); }
        if (traceback) { Py_DecRef(traceback); }

        JST_WARN("[RUNTIME_CONTEXT_PYTHON] Python cleanup helper '{}' failed during shutdown.", name);
        return;
    }

    Py_DecRef(result);
}

}  // namespace

Bridge::~Bridge() {
    (void)stop();
}

CPython::PyObject* Bridge::valueConverterTable() {
    if (valueConvertersFetched) {
        return valueConverters;
    }
    valueConvertersFetched = true;

    if (!globals) {
        return nullptr;
    }

    auto* factory = PyDict_GetItemString(globals, "_jetstream_value_converters");
    if (!factory || !PyCallable_Check(factory)) {
        return nullptr;
    }

    auto* table = PyObject_CallFunctionObjArgs(factory);
    if (!table) {
        (void)ClearPythonError();
        return nullptr;
    }

    valueConverters = table;
    return valueConverters;
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
                     const TensorMap& outputs,
                     const std::shared_ptr<Flowgraph::Environment>& environment,
                     const std::shared_ptr<Flowgraph::View>& view) {
    std::lock_guard<std::recursive_mutex> lifecycleLock(lifecycleMutex);

    const auto loadResult = Py_Load();
    if (loadResult != Result::SUCCESS) {
        setError("Runtime unavailable.");
        return loadResult;
    }

    consoleClear();
    JST_CHECK(stop());

    this->environment = environment;
    this->flowgraphView = view;

    GilScope gil;
    JST_CHECK(gil.result());

    globals = PyDict_New();
    if (!globals) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Can't create Python globals dictionary.");
        setError("Initialization error.");
        return Result::ERROR;
    }
    RegisterPythonGlobals();

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
    std::lock_guard<std::recursive_mutex> lifecycleLock(lifecycleMutex);

    if (!Py_IsLoaded()) {
        globals = nullptr;
        runner = nullptr;
        valueConverters = nullptr;
        valueConvertersFetched = false;
        environment.reset();
        environmentDict = nullptr;
        environmentSynced = false;
        flowgraphView.reset();
        metricsDict = nullptr;
        metricsRequests.clear();
        metricsSubscribeAll = false;
        inputAttributePorts.clear();
        outputAttributePorts.clear();
        return Result::SUCCESS;
    }

    GilScope gil;
    JST_CHECK(gil.result());

    environment.reset();
    flowgraphView.reset();
    destroyAttributePorts();
    destroyEnvironmentDict();
    destroyMetricsDict();

    if (valueConverters) {
        Py_DecRef(valueConverters);
        valueConverters = nullptr;
    }
    valueConvertersFetched = false;

    if (runner) {
        Py_DecRef(runner);
        runner = nullptr;
    }

    if (globals) {
        CallPythonShutdown(globals, "_jetstream_shutdown");
        if (UnregisterPythonGlobals()) {
            CallPythonShutdown(globals, "_jetstream_shutdown_runtime");
        }
        (void)consoleRefresh();
        Py_DecRef(globals);
        globals = nullptr;
    }

    return Result::SUCCESS;
}

Result Bridge::run() {
    std::lock_guard<std::recursive_mutex> lifecycleLock(lifecycleMutex);

    {
        std::lock_guard<std::mutex> lock(statusMutex);
        if (!healthy && status == kComputeErrorStatus) {
            return Result::SKIP;
        }
    }

    if (!runner || !globals) {
        const auto current = diagnostic();
        if (current.healthy || current.console.empty()) {
            JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Python compute entry has not been created.");
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
    JST_CHECK(gil.result());

    refreshAttributes();
    refreshEnvironment();
    refreshMetrics();

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
    flushAttributes();
    flushEnvironment();
    collectMetricsRequests();

    (void)consoleRefresh();
    Py_DecRef(result);
    setInfo("Running.");

    return Result::SUCCESS;
}

}  // namespace Jetstream
