#ifndef JETSTREAM_RUNTIME_PYTHON_BRIDGE_CPYTHON_BASE_HH
#define JETSTREAM_RUNTIME_PYTHON_BRIDGE_CPYTHON_BASE_HH

#include <cstdint>

#include "jetstream/types.hh"

namespace Jetstream::CPython {

using Py_ssize_t = std::intptr_t;

struct PyObject;
struct PyThreadState;

Result Py_Load();
bool Py_IsLoaded();

PyObject* PyDict_GetItemString(PyObject* dict, const char* key);
int PyCallable_Check(PyObject* object);
PyObject* PyObject_Str(PyObject* object);
PyObject* PyTuple_New(Py_ssize_t size);
int PyTuple_SetItem(PyObject* tuple, Py_ssize_t index, PyObject* item);
PyObject* PyLong_FromLongLong(long long value);
PyObject* PyMemoryView_FromMemory(char* mem, Py_ssize_t size, int flags);
Py_ssize_t PySequence_Size(PyObject* sequence);
PyObject* PySequence_GetItem(PyObject* sequence, Py_ssize_t index);
void Py_DecRef(PyObject* object);
void PyErr_Fetch(PyObject** type, PyObject** value, PyObject** traceback);
void PyErr_NormalizeException(PyObject** type, PyObject** value, PyObject** traceback);
PyObject* PyUnicode_FromString(const char* str);
const char* PyUnicode_AsUTF8(PyObject* object);
void Py_Initialize();
int Py_IsInitialized();
int PyGILState_Ensure();
void PyGILState_Release(int state);
PyThreadState* PyEval_SaveThread();
PyObject* PyRun_StringFlags(const char* str, int start, PyObject* globals, PyObject* locals, void* flags);
PyObject* PyDict_New();
PyObject* PyObject_GetAttrString(PyObject* object, const char* name);

PyObject* PyObject_CallFunctionObjArgs(PyObject* callable);
PyObject* PyObject_CallFunctionObjArgs(PyObject* callable, PyObject* arg0);
PyObject* PyObject_CallFunctionObjArgs(PyObject* callable, PyObject* arg0, PyObject* arg1);

}  // namespace Jetstream::CPython

#endif  // JETSTREAM_RUNTIME_PYTHON_BRIDGE_CPYTHON_BASE_HH
