#include "runtime/python/bridge/bootstrap/base.hh"

#include "jetstream/logger.hh"
#include "runtime/python/bridge/base.hh"
#include "runtime/python/bridge/bootstrap/activation.hh"
#include "runtime/python/bridge/convert.hh"
#include "runtime/python/bridge/cpython/base.hh"

namespace Jetstream {

using namespace CPython;

Result SwitchPythonEnvironmentPath(const std::string& previousPath,
                                   const std::string& currentPath) {
    std::lock_guard<std::recursive_mutex> operationLock(PythonOperationMutex());
    JST_CHECK(Py_Load());

    Bridge::Scope gil;
    JST_CHECK(gil.result());

    auto* globals = PyDict_New();
    if (!globals) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Can't create package switch globals.");
        return Result::ERROR;
    }

    auto* helperResult = PyRun_StringFlags(
        kPythonBootstrapActivation, 257, globals, globals, nullptr);
    if (!helperResult) {
        (void)ClearPythonError();
        Py_DecRef(globals);
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Can't initialize the package switcher.");
        return Result::ERROR;
    }
    Py_DecRef(helperResult);

    auto* switchPackages = PyDict_GetItemString(
        globals, "_jetstream_switch_packages");
    if (!switchPackages || !PyCallable_Check(switchPackages)) {
        Py_DecRef(globals);
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Package switch helper is unavailable.");
        return Result::ERROR;
    }

    auto* previous = PyUnicode_FromString(previousPath.c_str());
    auto* current = PyUnicode_FromString(currentPath.c_str());
    if (!previous || !current) {
        if (previous) { Py_DecRef(previous); }
        if (current) { Py_DecRef(current); }
        Py_DecRef(globals);
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Can't prepare package paths.");
        return Result::ERROR;
    }

    auto* result = PyObject_CallFunctionObjArgs(switchPackages, previous, current);
    Py_DecRef(previous);
    Py_DecRef(current);
    if (!result) {
        (void)ClearPythonError();
        Py_DecRef(globals);
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Can't switch the package path.");
        return Result::ERROR;
    }

    Py_DecRef(result);
    Py_DecRef(globals);
    return Result::SUCCESS;
}

}  // namespace Jetstream
