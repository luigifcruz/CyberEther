#include <cstddef>

#include "runtime/python/bridge/base.hh"
#include "runtime/python/bridge/cpython/base.hh"

#include "jetstream/logger.hh"

namespace Jetstream {

using namespace CPython;

namespace {

void DiscardPythonError() {
    PyObject* type = nullptr;
    PyObject* value = nullptr;
    PyObject* traceback = nullptr;
    PyErr_Fetch(&type, &value, &traceback);

    if (type) { Py_DecRef(type); }
    if (value) { Py_DecRef(value); }
    if (traceback) { Py_DecRef(traceback); }
}

std::string PythonToString(PyObject* object) {
    if (!object) {
        return {};
    }

    auto* str = PyObject_Str(object);
    if (!str) {
        DiscardPythonError();
        return {};
    }

    const char* chars = PyUnicode_AsUTF8(str);
    std::string result = chars ? chars : "";
    Py_DecRef(str);

    return result;
}

std::string TakePythonError(PyObject* globals) {
    PyObject* type = nullptr;
    PyObject* value = nullptr;
    PyObject* traceback = nullptr;
    PyErr_Fetch(&type, &value, &traceback);

    if (!type && !value && !traceback) {
        return {};
    }

    PyErr_NormalizeException(&type, &value, &traceback);

    if (globals && type && traceback) {
        auto* formatter = PyDict_GetItemString(globals, "_jetstream_format_exception");
        if (formatter && PyCallable_Check(formatter)) {
            auto* formatted = PyObject_CallFunctionObjArgs(formatter, type, value, traceback);
            if (formatted) {
                const auto message = PythonToString(formatted);
                Py_DecRef(formatted);
                if (!message.empty()) {
                    Py_DecRef(type);
                    if (value) { Py_DecRef(value); }
                    Py_DecRef(traceback);
                    return message;
                }
            } else {
                DiscardPythonError();
            }
        }
    }

    std::string typeName = "PythonError";
    if (type) {
        auto* name = PyObject_GetAttrString(type, "__name__");
        const auto converted = PythonToString(name);
        if (!converted.empty()) {
            typeName = converted;
        }
        if (name) {
            Py_DecRef(name);
        } else {
            DiscardPythonError();
        }
    }

    const std::string message = PythonToString(value);

    if (type) { Py_DecRef(type); }
    if (value) { Py_DecRef(value); }
    if (traceback) { Py_DecRef(traceback); }

    if (message.empty()) {
        return typeName;
    }

    return typeName + ": " + message;
}

}  // namespace

void Bridge::setInfo(const std::string& text) {
    std::lock_guard<std::mutex> lock(statusMutex);
    healthy = true;
    status = text;
}

void Bridge::setError(const std::string& text, std::string details) {
    if (details.empty() && Py_IsLoaded()) {
        details = TakePythonError(globals);
    }
    if (details.empty()) {
        details = JST_LOG_LAST_ERROR();
    }

    const bool refreshed = (Py_IsLoaded() && globals) && consoleRefresh();
    if (!refreshed && !details.empty()) {
        consoleClear();
    }

    if (!details.empty()) {
        consoleAppend(details);
    }

    bool repeated = false;
    {
        std::lock_guard<std::mutex> lock(statusMutex);
        repeated = !healthy && status == text;
        healthy = false;
        status = text;
    }

    if (!repeated && !details.empty()) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] {}\n{}", text, details);
    }
}

}  // namespace Jetstream
