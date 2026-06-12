#include <dlfcn.h>
#include <mutex>

#include "runtime/python/bridge/cpython/base.hh"

#include "jetstream/logger.hh"

namespace Jetstream::CPython {

namespace {

constexpr const char* kPythonLibraryPath = "/opt/homebrew/opt/python@3.14/Frameworks/Python.framework/Versions/3.14/Python";

std::mutex& LoaderMutex() {
    static std::mutex mutex;
    return mutex;
}

template<typename T>
Result LoadSymbol(void* handle, T& target, const char* symbol) {
    target = reinterpret_cast<T>(dlsym(handle, symbol));
    if (!target) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Can't load Python symbol '{}'.", symbol);
        return Result::ERROR;
    }
    return Result::SUCCESS;
}

using PyInitializeFn = void (*)();
using PyIsInitializedFn = int (*)();
using PyGILStateEnsureFn = int (*)();
using PyGILStateReleaseFn = void (*)(int);
using PyEvalSaveThreadFn = PyThreadState* (*)();
using PyRunStringFlagsFn = PyObject* (*)(const char*, int, PyObject*, PyObject*, void*);
using PyDictNewFn = PyObject* (*)();
using PyDictGetItemStringFn = PyObject* (*)(PyObject*, const char*);
using PyCallableCheckFn = int (*)(PyObject*);
using PyObjectGetAttrStringFn = PyObject* (*)(PyObject*, const char*);
using PyObjectCallFunctionObjArgsFn = PyObject* (*)(PyObject*, ...);
using PyObjectStrFn = PyObject* (*)(PyObject*);
using PyTupleNewFn = PyObject* (*)(Py_ssize_t);
using PyTupleSetItemFn = int (*)(PyObject*, Py_ssize_t, PyObject*);
using PyLongFromLongLongFn = PyObject* (*)(long long);
using PyMemoryViewFromMemoryFn = PyObject* (*)(char*, Py_ssize_t, int);
using PySequenceSizeFn = Py_ssize_t (*)(PyObject*);
using PySequenceGetItemFn = PyObject* (*)(PyObject*, Py_ssize_t);
using PyDecRefFn = void (*)(PyObject*);
using PyErrFetchFn = void (*)(PyObject**, PyObject**, PyObject**);
using PyErrNormalizeExceptionFn = void (*)(PyObject**, PyObject**, PyObject**);
using PyUnicodeFromStringFn = PyObject* (*)(const char*);
using PyUnicodeAsUTF8Fn = const char* (*)(PyObject*);

struct PythonApi {
    PyInitializeFn Py_Initialize = nullptr;
    PyIsInitializedFn Py_IsInitialized = nullptr;
    PyGILStateEnsureFn PyGILState_Ensure = nullptr;
    PyGILStateReleaseFn PyGILState_Release = nullptr;
    PyEvalSaveThreadFn PyEval_SaveThread = nullptr;
    PyRunStringFlagsFn PyRun_StringFlags = nullptr;
    PyDictNewFn PyDict_New = nullptr;
    PyDictGetItemStringFn PyDict_GetItemString = nullptr;
    PyCallableCheckFn PyCallable_Check = nullptr;
    PyObjectGetAttrStringFn PyObject_GetAttrString = nullptr;
    PyObjectCallFunctionObjArgsFn PyObject_CallFunctionObjArgs = nullptr;
    PyObjectStrFn PyObject_Str = nullptr;
    PyTupleNewFn PyTuple_New = nullptr;
    PyTupleSetItemFn PyTuple_SetItem = nullptr;
    PyLongFromLongLongFn PyLong_FromLongLong = nullptr;
    PyMemoryViewFromMemoryFn PyMemoryView_FromMemory = nullptr;
    PySequenceSizeFn PySequence_Size = nullptr;
    PySequenceGetItemFn PySequence_GetItem = nullptr;
    PyDecRefFn Py_DecRef = nullptr;
    PyErrFetchFn PyErr_Fetch = nullptr;
    PyErrNormalizeExceptionFn PyErr_NormalizeException = nullptr;
    PyUnicodeFromStringFn PyUnicode_FromString = nullptr;
    PyUnicodeAsUTF8Fn PyUnicode_AsUTF8 = nullptr;
};

Result LoadSymbols(void* handle, PythonApi& api) {
    JST_CHECK(LoadSymbol(handle, api.Py_Initialize, "Py_Initialize"));
    JST_CHECK(LoadSymbol(handle, api.Py_IsInitialized, "Py_IsInitialized"));
    JST_CHECK(LoadSymbol(handle, api.PyGILState_Ensure, "PyGILState_Ensure"));
    JST_CHECK(LoadSymbol(handle, api.PyGILState_Release, "PyGILState_Release"));
    JST_CHECK(LoadSymbol(handle, api.PyEval_SaveThread, "PyEval_SaveThread"));
    JST_CHECK(LoadSymbol(handle, api.PyRun_StringFlags, "PyRun_StringFlags"));
    JST_CHECK(LoadSymbol(handle, api.PyDict_New, "PyDict_New"));
    JST_CHECK(LoadSymbol(handle, api.PyDict_GetItemString, "PyDict_GetItemString"));
    JST_CHECK(LoadSymbol(handle, api.PyCallable_Check, "PyCallable_Check"));
    JST_CHECK(LoadSymbol(handle, api.PyObject_GetAttrString, "PyObject_GetAttrString"));
    JST_CHECK(LoadSymbol(handle, api.PyObject_CallFunctionObjArgs, "PyObject_CallFunctionObjArgs"));
    JST_CHECK(LoadSymbol(handle, api.PyObject_Str, "PyObject_Str"));
    JST_CHECK(LoadSymbol(handle, api.PyTuple_New, "PyTuple_New"));
    JST_CHECK(LoadSymbol(handle, api.PyTuple_SetItem, "PyTuple_SetItem"));
    JST_CHECK(LoadSymbol(handle, api.PyLong_FromLongLong, "PyLong_FromLongLong"));
    JST_CHECK(LoadSymbol(handle, api.PyMemoryView_FromMemory, "PyMemoryView_FromMemory"));
    JST_CHECK(LoadSymbol(handle, api.PySequence_Size, "PySequence_Size"));
    JST_CHECK(LoadSymbol(handle, api.PySequence_GetItem, "PySequence_GetItem"));
    JST_CHECK(LoadSymbol(handle, api.Py_DecRef, "Py_DecRef"));
    JST_CHECK(LoadSymbol(handle, api.PyErr_Fetch, "PyErr_Fetch"));
    JST_CHECK(LoadSymbol(handle, api.PyErr_NormalizeException, "PyErr_NormalizeException"));
    JST_CHECK(LoadSymbol(handle, api.PyUnicode_FromString, "PyUnicode_FromString"));
    JST_CHECK(LoadSymbol(handle, api.PyUnicode_AsUTF8, "PyUnicode_AsUTF8"));

    return Result::SUCCESS;
}

PythonApi s_api;
void* s_libraryHandle = nullptr;
bool s_libraryLoaded = false;

}  // namespace

PyObject* PyObject_CallFunctionObjArgs(PyObject* callable) {
    return s_api.PyObject_CallFunctionObjArgs(callable, static_cast<PyObject*>(nullptr));
}

PyObject* PyObject_CallFunctionObjArgs(PyObject* callable, PyObject* arg0) {
    return s_api.PyObject_CallFunctionObjArgs(callable, arg0, static_cast<PyObject*>(nullptr));
}

PyObject* PyObject_CallFunctionObjArgs(PyObject* callable, PyObject* arg0, PyObject* arg1) {
    return s_api.PyObject_CallFunctionObjArgs(callable, arg0, arg1, static_cast<PyObject*>(nullptr));
}

Result Py_Load() {
    std::lock_guard<std::mutex> lock(LoaderMutex());

    if (s_libraryLoaded) {
        return Result::SUCCESS;
    }

    auto* handle = dlopen(kPythonLibraryPath, RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Can't load Python library '{}': {}.",
                  kPythonLibraryPath, dlerror());
        return Result::ERROR;
    }

    PythonApi api;
    const auto symbolsResult = LoadSymbols(handle, api);
    if (symbolsResult != Result::SUCCESS) {
        dlclose(handle);
        return symbolsResult;
    }

    if (!api.Py_IsInitialized()) {
        api.Py_Initialize();
        api.PyEval_SaveThread();
    }

    s_api = api;
    s_libraryHandle = handle;
    s_libraryLoaded = true;
    return Result::SUCCESS;
}

bool Py_IsLoaded() {
    std::lock_guard<std::mutex> lock(LoaderMutex());
    return s_libraryLoaded;
}

PyObject* PyDict_GetItemString(PyObject* dict, const char* key) {
    return s_api.PyDict_GetItemString(dict, key);
}

int PyCallable_Check(PyObject* object) {
    return s_api.PyCallable_Check(object);
}

PyObject* PyObject_Str(PyObject* object) {
    return s_api.PyObject_Str(object);
}

PyObject* PyTuple_New(Py_ssize_t size) {
    return s_api.PyTuple_New(size);
}

int PyTuple_SetItem(PyObject* tuple, Py_ssize_t index, PyObject* item) {
    return s_api.PyTuple_SetItem(tuple, index, item);
}

PyObject* PyLong_FromLongLong(long long value) {
    return s_api.PyLong_FromLongLong(value);
}

PyObject* PyMemoryView_FromMemory(char* mem, Py_ssize_t size, int flags) {
    return s_api.PyMemoryView_FromMemory(mem, size, flags);
}

Py_ssize_t PySequence_Size(PyObject* sequence) {
    return s_api.PySequence_Size(sequence);
}

PyObject* PySequence_GetItem(PyObject* sequence, Py_ssize_t index) {
    return s_api.PySequence_GetItem(sequence, index);
}

void Py_DecRef(PyObject* object) {
    s_api.Py_DecRef(object);
}

void PyErr_Fetch(PyObject** type, PyObject** value, PyObject** traceback) {
    s_api.PyErr_Fetch(type, value, traceback);
}

void PyErr_NormalizeException(PyObject** type, PyObject** value, PyObject** traceback) {
    s_api.PyErr_NormalizeException(type, value, traceback);
}

PyObject* PyUnicode_FromString(const char* str) {
    return s_api.PyUnicode_FromString(str);
}

const char* PyUnicode_AsUTF8(PyObject* object) {
    return s_api.PyUnicode_AsUTF8(object);
}

void Py_Initialize() {
    s_api.Py_Initialize();
}

int Py_IsInitialized() {
    return s_api.Py_IsInitialized();
}

int PyGILState_Ensure() {
    return s_api.PyGILState_Ensure();
}

void PyGILState_Release(int state) {
    s_api.PyGILState_Release(state);
}

PyThreadState* PyEval_SaveThread() {
    return s_api.PyEval_SaveThread();
}

PyObject* PyRun_StringFlags(const char* str, int start, PyObject* globals, PyObject* locals, void* flags) {
    return s_api.PyRun_StringFlags(str, start, globals, locals, flags);
}

PyObject* PyDict_New() {
    return s_api.PyDict_New();
}

PyObject* PyObject_GetAttrString(PyObject* object, const char* name) {
    return s_api.PyObject_GetAttrString(object, name);
}

}  // namespace Jetstream::CPython
