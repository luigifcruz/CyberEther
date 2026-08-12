#include "runtime/python/bridge/base.hh"
#include "runtime/python/bridge/convert.hh"

#include "jetstream/flowgraph_view.hh"
#include "jetstream/logger.hh"

namespace Jetstream {

using namespace CPython;

namespace {

constexpr const char* kMetricsSubscribeAllRequest = ".jetstream.metrics.subscribe_all";

}  // namespace

PyObject* Bridge::createMetricsDict() {
    if (!globals) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Can't create Python metrics without globals.");
        return nullptr;
    }

    auto* factory = PyDict_GetItemString(globals, "_jetstream_create_metrics");
    if (!factory || !PyCallable_Check(factory)) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Python runtime helper '_jetstream_create_metrics' is unavailable.");
        return nullptr;
    }

    auto* dict = PyObject_CallFunctionObjArgs(factory);
    if (!dict) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Can't create Python metrics dict.");
        return nullptr;
    }

    Py_IncRef(dict);
    metricsDict = dict;

    return dict;
}

void Bridge::refreshMetrics() {
    if (!metricsDict || !flowgraphView) {
        return;
    }

    std::unordered_set<std::string> blocks = metricsRequests;
    if (metricsSubscribeAll) {
        std::vector<std::string> keys;
        if (flowgraphView->keys(keys) == Result::SUCCESS) {
            PyDict_Clear(metricsDict);
            for (const auto& key : keys) {
                blocks.insert(key);
            }
        }
    }

    if (blocks.empty()) {
        return;
    }

    for (const auto& block : blocks) {
        auto* values = PyDict_New();
        if (!values) {
            (void)ClearPythonError();
            continue;
        }

        if (flowgraphView->has(block)) {
            std::vector<Flowgraph::View::MetricEntry> entries;
            if (flowgraphView->metrics(block, entries) == Result::SUCCESS) {
                for (const auto& entry : entries) {
                    if (entry.format.starts_with("private-")) {
                        continue;
                    }

                    auto* object = AnyToPyObject(entry.value, valueConverterTable());
                    if (!object) {
                        JST_TRACE("[RUNTIME_CONTEXT_PYTHON] Skipping unsupported metric '{}:{}'.",
                                  block, entry.name);
                        (void)ClearPythonError();
                        continue;
                    }

                    if (PyDict_SetItemString(values, entry.name.c_str(), object) != 0) {
                        (void)ClearPythonError();
                    }
                    Py_DecRef(object);
                }
            }
        }

        if (PyDict_SetItemString(metricsDict, block.c_str(), values) != 0) {
            (void)ClearPythonError();
        }
        Py_DecRef(values);
    }
}

void Bridge::collectMetricsRequests() {
    if (!metricsDict || !globals) {
        return;
    }

    auto* consume = PyDict_GetItemString(globals, "_jetstream_consume_metrics_requests");
    if (!consume || !PyCallable_Check(consume)) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Python runtime helper '_jetstream_consume_metrics_requests' is unavailable.");
        return;
    }

    auto* requests = PyObject_CallFunctionObjArgs(consume, metricsDict);
    if (!requests) {
        (void)ClearPythonError();
        return;
    }

    const auto count = PySequence_Size(requests);
    for (Py_ssize_t index = 0; index < count; ++index) {
        auto* key = PySequence_GetItem(requests, index);
        if (!key) {
            (void)ClearPythonError();
            continue;
        }

        const char* keyStr = PyUnicode_AsUTF8(key);
        if (keyStr) {
            if (std::string(keyStr) == kMetricsSubscribeAllRequest) {
                metricsSubscribeAll = true;
            } else {
                metricsRequests.insert(keyStr);
            }
        } else {
            (void)ClearPythonError();
        }
        Py_DecRef(key);
    }

    Py_DecRef(requests);
}

void Bridge::destroyMetricsDict() {
    if (metricsDict) {
        Py_DecRef(metricsDict);
        metricsDict = nullptr;
    }
    metricsRequests.clear();
    metricsSubscribeAll = false;
}

}  // namespace Jetstream
