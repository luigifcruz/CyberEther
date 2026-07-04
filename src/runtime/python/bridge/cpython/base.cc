#include <mutex>
#include <string>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#ifdef ERROR
#undef ERROR
#endif
#ifdef FATAL
#undef FATAL
#endif
#else
#include <dlfcn.h>
#endif

#include "runtime/python/bridge/cpython/base.hh"

#include "jetstream/backend/base.hh"
#include "jetstream/logger.hh"
#include "jetstream/runtime_context_python.hh"

namespace Jetstream::CPython {

namespace {

std::mutex& LoaderMutex() {
    static std::mutex mutex;
    return mutex;
}

template<typename T>
Result LoadSymbol(void* handle, T& target, const char* symbol) {
#if defined(_WIN32)
    target = reinterpret_cast<T>(GetProcAddress(reinterpret_cast<HMODULE>(handle), symbol));
#else
    target = reinterpret_cast<T>(dlsym(handle, symbol));
#endif
    if (!target) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Can't load Python symbol '{}'.", symbol);
        return Result::ERROR;
    }
    return Result::SUCCESS;
}

template<typename T>
Result LoadOptionalSymbol(void* handle, T& target, const char* symbol) {
#if defined(_WIN32)
    target = reinterpret_cast<T>(GetProcAddress(reinterpret_cast<HMODULE>(handle), symbol));
#else
    target = reinterpret_cast<T>(dlsym(handle, symbol));
#endif
    return Result::SUCCESS;
}

void CloseLibrary(void* handle) {
    if (!handle) {
        return;
    }

#if defined(_WIN32)
    FreeLibrary(reinterpret_cast<HMODULE>(handle));
#else
    dlclose(handle);
#endif
}

std::string ConfiguredPythonPath() {
#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
    return Backend::State<DeviceType::CPU>()->getPythonRuntimePath();
#else
    return {};
#endif
}

bool TryOpenLibrary(const std::string& path, void*& handle, std::string& error) {
#if defined(_WIN32)
    handle = reinterpret_cast<void*>(LoadLibraryA(path.c_str()));
    if (handle) {
        return true;
    }

    error = "Windows error " + std::to_string(GetLastError());
    return false;
#else
    dlerror();
    handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (handle) {
        return true;
    }

    if (const char* dlError = dlerror()) {
        error = dlError;
    } else {
        error = "unknown dlopen error";
    }
    return false;
#endif
}

Result OpenPythonLibrary(PythonRuntimeContext::Validation& validation, void*& handle) {
    validation = PythonRuntimeContext::ValidateRuntimePath(ConfiguredPythonPath());
    if (!validation.valid) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] {}", validation.message);
        for (const auto& attempt : validation.attempts) {
            JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Attempted {}", attempt);
        }
        return Result::ERROR;
    }

    std::string error;
    if (!TryOpenLibrary(validation.libraryPath, handle, error)) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Can't load Python library '{}': {}.",
                  validation.libraryPath,
                  error);
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

using PyInitializeFn = void (*)();
using PyIsInitializedFn = int (*)();
using PySetProgramNameFn = void (*)(const wchar_t*);
using PyDecodeLocaleFn = wchar_t* (*)(const char*, size_t*);
using PyGILStateEnsureFn = int (*)();
using PyGILStateReleaseFn = void (*)(int);
using PyInterpreterStateMainFn = PyInterpreterState* (*)();
using PyThreadStateNewFn = PyThreadState* (*)(PyInterpreterState*);
using PyEvalRestoreThreadFn = void (*)(PyThreadState*);
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
using PyLongFromUnsignedLongLongFn = PyObject* (*)(unsigned long long);
using PyLongAsLongLongFn = long long (*)(PyObject*);
using PyLongAsUnsignedLongLongFn = unsigned long long (*)(PyObject*);
using PyNumberIndexFn = PyObject* (*)(PyObject*);
using PyObjectIsTrueFn = int (*)(PyObject*);
using PyFloatFromDoubleFn = PyObject* (*)(double);
using PyFloatAsDoubleFn = double (*)(PyObject*);
using PyComplexFromDoublesFn = PyObject* (*)(double, double);
using PyComplexAsCComplexFn = PyComplexValue (*)(PyObject*);
using PyBoolFromLongFn = PyObject* (*)(long);
using PyDictSetItemStringFn = int (*)(PyObject*, const char*, PyObject*);
using PyDictNextFn = int (*)(PyObject*, Py_ssize_t*, PyObject**, PyObject**);
using PyDictDelItemStringFn = int (*)(PyObject*, const char*);
using PyDictClearFn = void (*)(PyObject*);
using PyIncRefFn = void (*)(PyObject*);
using PyMemoryViewFromMemoryFn = PyObject* (*)(char*, Py_ssize_t, int);
using PySequenceSizeFn = Py_ssize_t (*)(PyObject*);
using PySequenceGetItemFn = PyObject* (*)(PyObject*, Py_ssize_t);
using PyDecRefFn = void (*)(PyObject*);
using PyErrFetchFn = void (*)(PyObject**, PyObject**, PyObject**);
using PyErrNormalizeExceptionFn = void (*)(PyObject**, PyObject**, PyObject**);
using PyUnicodeFromStringFn = PyObject* (*)(const char*);
using PyUnicodeAsUTF8Fn = const char* (*)(PyObject*);
using PyBytesFromStringAndSizeFn = PyObject* (*)(const char*, Py_ssize_t);
using PyBytesAsStringAndSizeFn = int (*)(PyObject*, char**, Py_ssize_t*);

struct PythonApi {
    PyInitializeFn Py_Initialize = nullptr;
    PyIsInitializedFn Py_IsInitialized = nullptr;
    PySetProgramNameFn Py_SetProgramName = nullptr;
    PyDecodeLocaleFn Py_DecodeLocale = nullptr;
    PyGILStateEnsureFn PyGILState_Ensure = nullptr;
    PyGILStateReleaseFn PyGILState_Release = nullptr;
    PyInterpreterStateMainFn PyInterpreterState_Main = nullptr;
    PyThreadStateNewFn PyThreadState_New = nullptr;
    PyEvalRestoreThreadFn PyEval_RestoreThread = nullptr;
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
    PyLongFromUnsignedLongLongFn PyLong_FromUnsignedLongLong = nullptr;
    PyLongAsLongLongFn PyLong_AsLongLong = nullptr;
    PyLongAsUnsignedLongLongFn PyLong_AsUnsignedLongLong = nullptr;
    PyNumberIndexFn PyNumber_Index = nullptr;
    PyObjectIsTrueFn PyObject_IsTrue = nullptr;
    PyFloatFromDoubleFn PyFloat_FromDouble = nullptr;
    PyFloatAsDoubleFn PyFloat_AsDouble = nullptr;
    PyComplexFromDoublesFn PyComplex_FromDoubles = nullptr;
    PyComplexAsCComplexFn PyComplex_AsCComplex = nullptr;
    PyBoolFromLongFn PyBool_FromLong = nullptr;
    PyObject* Py_NoneStruct = nullptr;
    PyDictSetItemStringFn PyDict_SetItemString = nullptr;
    PyDictNextFn PyDict_Next = nullptr;
    PyDictDelItemStringFn PyDict_DelItemString = nullptr;
    PyDictClearFn PyDict_Clear = nullptr;
    PyIncRefFn Py_IncRef = nullptr;
    PyMemoryViewFromMemoryFn PyMemoryView_FromMemory = nullptr;
    PySequenceSizeFn PySequence_Size = nullptr;
    PySequenceGetItemFn PySequence_GetItem = nullptr;
    PyDecRefFn Py_DecRef = nullptr;
    PyErrFetchFn PyErr_Fetch = nullptr;
    PyErrNormalizeExceptionFn PyErr_NormalizeException = nullptr;
    PyUnicodeFromStringFn PyUnicode_FromString = nullptr;
    PyUnicodeAsUTF8Fn PyUnicode_AsUTF8 = nullptr;
    PyBytesFromStringAndSizeFn PyBytes_FromStringAndSize = nullptr;
    PyBytesAsStringAndSizeFn PyBytes_AsStringAndSize = nullptr;
};

Result LoadSymbols(void* handle, PythonApi& api) {
    JST_CHECK(LoadSymbol(handle, api.Py_Initialize, "Py_Initialize"));
    JST_CHECK(LoadSymbol(handle, api.Py_IsInitialized, "Py_IsInitialized"));
    JST_CHECK(LoadOptionalSymbol(handle, api.Py_SetProgramName, "Py_SetProgramName"));
    JST_CHECK(LoadOptionalSymbol(handle, api.Py_DecodeLocale, "Py_DecodeLocale"));
    JST_CHECK(LoadSymbol(handle, api.PyGILState_Ensure, "PyGILState_Ensure"));
    JST_CHECK(LoadSymbol(handle, api.PyGILState_Release, "PyGILState_Release"));
    JST_CHECK(LoadSymbol(handle, api.PyInterpreterState_Main, "PyInterpreterState_Main"));
    JST_CHECK(LoadSymbol(handle, api.PyThreadState_New, "PyThreadState_New"));
    JST_CHECK(LoadSymbol(handle, api.PyEval_RestoreThread, "PyEval_RestoreThread"));
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
    JST_CHECK(LoadSymbol(handle, api.PyLong_FromUnsignedLongLong, "PyLong_FromUnsignedLongLong"));
    JST_CHECK(LoadSymbol(handle, api.PyLong_AsLongLong, "PyLong_AsLongLong"));
    JST_CHECK(LoadSymbol(handle, api.PyLong_AsUnsignedLongLong, "PyLong_AsUnsignedLongLong"));
    JST_CHECK(LoadSymbol(handle, api.PyNumber_Index, "PyNumber_Index"));
    JST_CHECK(LoadSymbol(handle, api.PyObject_IsTrue, "PyObject_IsTrue"));
    JST_CHECK(LoadSymbol(handle, api.PyFloat_FromDouble, "PyFloat_FromDouble"));
    JST_CHECK(LoadSymbol(handle, api.PyFloat_AsDouble, "PyFloat_AsDouble"));
    JST_CHECK(LoadSymbol(handle, api.PyComplex_FromDoubles, "PyComplex_FromDoubles"));
    JST_CHECK(LoadSymbol(handle, api.PyComplex_AsCComplex, "PyComplex_AsCComplex"));
    JST_CHECK(LoadSymbol(handle, api.PyBool_FromLong, "PyBool_FromLong"));
    JST_CHECK(LoadSymbol(handle, api.Py_NoneStruct, "_Py_NoneStruct"));
    JST_CHECK(LoadSymbol(handle, api.PyDict_SetItemString, "PyDict_SetItemString"));
    JST_CHECK(LoadSymbol(handle, api.PyDict_Next, "PyDict_Next"));
    JST_CHECK(LoadSymbol(handle, api.PyDict_DelItemString, "PyDict_DelItemString"));
    JST_CHECK(LoadSymbol(handle, api.PyDict_Clear, "PyDict_Clear"));
    JST_CHECK(LoadSymbol(handle, api.Py_IncRef, "Py_IncRef"));
    JST_CHECK(LoadSymbol(handle, api.PyMemoryView_FromMemory, "PyMemoryView_FromMemory"));
    JST_CHECK(LoadSymbol(handle, api.PySequence_Size, "PySequence_Size"));
    JST_CHECK(LoadSymbol(handle, api.PySequence_GetItem, "PySequence_GetItem"));
    JST_CHECK(LoadSymbol(handle, api.Py_DecRef, "Py_DecRef"));
    JST_CHECK(LoadSymbol(handle, api.PyErr_Fetch, "PyErr_Fetch"));
    JST_CHECK(LoadSymbol(handle, api.PyErr_NormalizeException, "PyErr_NormalizeException"));
    JST_CHECK(LoadSymbol(handle, api.PyUnicode_FromString, "PyUnicode_FromString"));
    JST_CHECK(LoadSymbol(handle, api.PyUnicode_AsUTF8, "PyUnicode_AsUTF8"));
    JST_CHECK(LoadSymbol(handle, api.PyBytes_FromStringAndSize, "PyBytes_FromStringAndSize"));
    JST_CHECK(LoadSymbol(handle, api.PyBytes_AsStringAndSize, "PyBytes_AsStringAndSize"));

    return Result::SUCCESS;
}

PythonApi s_api;
void* s_libraryHandle = nullptr;
bool s_libraryLoaded = false;
PyInterpreterState* s_interpreter = nullptr;
wchar_t* s_programName = nullptr;

void SetPythonProgramName(const PythonApi& api, const std::string& programPath) {
    if (programPath.empty() || s_programName || !api.Py_SetProgramName || !api.Py_DecodeLocale) {
        return;
    }

    s_programName = api.Py_DecodeLocale(programPath.c_str(), nullptr);
    if (!s_programName) {
        JST_WARN("[RUNTIME_CONTEXT_PYTHON] Can't decode Python program path '{}'.", programPath);
        return;
    }

    api.Py_SetProgramName(s_programName);
}

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

PyObject* PyObject_CallFunctionObjArgs(PyObject* callable, PyObject* arg0, PyObject* arg1,
                                       PyObject* arg2) {
    return s_api.PyObject_CallFunctionObjArgs(callable, arg0, arg1, arg2,
                                              static_cast<PyObject*>(nullptr));
}

PyObject* PyObject_CallFunctionObjArgs(PyObject* callable, PyObject* arg0, PyObject* arg1,
                                       PyObject* arg2, PyObject* arg3) {
    return s_api.PyObject_CallFunctionObjArgs(callable, arg0, arg1, arg2, arg3,
                                              static_cast<PyObject*>(nullptr));
}

PyObject* PyObject_CallFunctionObjArgs(PyObject* callable, PyObject* arg0, PyObject* arg1,
                                       PyObject* arg2, PyObject* arg3, PyObject* arg4) {
    return s_api.PyObject_CallFunctionObjArgs(callable, arg0, arg1, arg2, arg3, arg4,
                                              static_cast<PyObject*>(nullptr));
}

PyObject* PyObject_CallFunctionObjArgs(PyObject* callable, PyObject* arg0, PyObject* arg1,
                                       PyObject* arg2, PyObject* arg3, PyObject* arg4,
                                       PyObject* arg5) {
    return s_api.PyObject_CallFunctionObjArgs(callable, arg0, arg1, arg2, arg3, arg4, arg5,
                                              static_cast<PyObject*>(nullptr));
}

Result Py_Load() {
    std::lock_guard<std::mutex> lock(LoaderMutex());

    if (s_libraryLoaded) {
        return Result::SUCCESS;
    }

    PythonRuntimeContext::Validation validation;
    void* handle = nullptr;
    JST_CHECK(OpenPythonLibrary(validation, handle));

    PythonApi api;
    const auto symbolsResult = LoadSymbols(handle, api);
    if (symbolsResult != Result::SUCCESS) {
        CloseLibrary(handle);
        return symbolsResult;
    }

    const bool initializedHere = !api.Py_IsInitialized();
    if (initializedHere) {
        SetPythonProgramName(api, validation.programPath);
        api.Py_Initialize();
    }

    int gilState = 0;
    if (!initializedHere) {
        gilState = api.PyGILState_Ensure();
    }

    auto* interpreter = api.PyInterpreterState_Main();
    if (!interpreter) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Can't access Python main interpreter state.");
        if (!initializedHere) {
            api.PyGILState_Release(gilState);
        }
        CloseLibrary(handle);
        return Result::ERROR;
    }

    if (initializedHere) {
        api.PyEval_SaveThread();
    } else {
        api.PyGILState_Release(gilState);
    }

    s_api = api;
    s_libraryHandle = handle;
    s_libraryLoaded = true;
    s_interpreter = interpreter;
    JST_INFO("[RUNTIME_CONTEXT_PYTHON] Loaded Python library '{}'.", validation.libraryPath);
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

PyObject* PyLong_FromUnsignedLongLong(unsigned long long value) {
    return s_api.PyLong_FromUnsignedLongLong(value);
}

long long PyLong_AsLongLong(PyObject* object) {
    return s_api.PyLong_AsLongLong(object);
}

unsigned long long PyLong_AsUnsignedLongLong(PyObject* object) {
    return s_api.PyLong_AsUnsignedLongLong(object);
}

PyObject* PyNumber_Index(PyObject* object) {
    return s_api.PyNumber_Index(object);
}

int PyObject_IsTrue(PyObject* object) {
    return s_api.PyObject_IsTrue(object);
}

PyObject* PyFloat_FromDouble(double value) {
    return s_api.PyFloat_FromDouble(value);
}

double PyFloat_AsDouble(PyObject* object) {
    return s_api.PyFloat_AsDouble(object);
}

PyObject* PyComplex_FromDoubles(double real, double imag) {
    return s_api.PyComplex_FromDoubles(real, imag);
}

PyComplexValue PyComplex_AsCComplex(PyObject* object) {
    return s_api.PyComplex_AsCComplex(object);
}

PyObject* PyBool_FromLong(long value) {
    return s_api.PyBool_FromLong(value);
}

PyObject* Py_GetNone() {
    return s_api.Py_NoneStruct;
}

int PyDict_SetItemString(PyObject* dict, const char* key, PyObject* value) {
    return s_api.PyDict_SetItemString(dict, key, value);
}

int PyDict_Next(PyObject* dict, Py_ssize_t* pos, PyObject** key, PyObject** value) {
    return s_api.PyDict_Next(dict, pos, key, value);
}

int PyDict_DelItemString(PyObject* dict, const char* key) {
    return s_api.PyDict_DelItemString(dict, key);
}

void PyDict_Clear(PyObject* dict) {
    s_api.PyDict_Clear(dict);
}

void Py_IncRef(PyObject* object) {
    s_api.Py_IncRef(object);
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

PyObject* PyBytes_FromStringAndSize(const char* data, Py_ssize_t size) {
    return s_api.PyBytes_FromStringAndSize(data, size);
}

int PyBytes_AsStringAndSize(PyObject* object, char** data, Py_ssize_t* size) {
    return s_api.PyBytes_AsStringAndSize(object, data, size);
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

PyThreadState* PyThreadState_New() {
    return s_interpreter ? s_api.PyThreadState_New(s_interpreter) : nullptr;
}

void PyEval_RestoreThread(PyThreadState* state) {
    s_api.PyEval_RestoreThread(state);
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
