#ifndef JETSTREAM_RUNTIME_PYTHON_BRIDGE_CPYTHON_BASE_HH
#define JETSTREAM_RUNTIME_PYTHON_BRIDGE_CPYTHON_BASE_HH

#include <cstdint>

#include "jetstream/types.hh"

namespace Jetstream::CPython {

using Py_ssize_t = std::intptr_t;

struct PyObject;
struct PyInterpreterState;
struct PyThreadState;

Result Py_Load();
bool Py_IsLoaded();

PyObject* PyDict_GetItemString(PyObject* dict, const char* key);
int PyCallable_Check(PyObject* object);
PyObject* PyObject_Str(PyObject* object);
PyObject* PyTuple_New(Py_ssize_t size);
int PyTuple_SetItem(PyObject* tuple, Py_ssize_t index, PyObject* item);
PyObject* PyLong_FromLongLong(long long value);
PyObject* PyLong_FromUnsignedLongLong(unsigned long long value);
long long PyLong_AsLongLong(PyObject* object);
unsigned long long PyLong_AsUnsignedLongLong(PyObject* object);
PyObject* PyNumber_Index(PyObject* object);
int PyObject_IsTrue(PyObject* object);
PyObject* PyFloat_FromDouble(double value);
double PyFloat_AsDouble(PyObject* object);
struct PyComplexValue {
    double real;
    double imag;
};

PyObject* PyComplex_FromDoubles(double real, double imag);
PyComplexValue PyComplex_AsCComplex(PyObject* object);
PyObject* PyBool_FromLong(long value);
int PyDict_SetItemString(PyObject* dict, const char* key, PyObject* value);
int PyDict_DelItemString(PyObject* dict, const char* key);
void PyDict_Clear(PyObject* dict);
int PyDict_Next(PyObject* dict, Py_ssize_t* pos, PyObject** key, PyObject** value);
void Py_IncRef(PyObject* object);
PyObject* PyMemoryView_FromMemory(char* mem, Py_ssize_t size, int flags);
Py_ssize_t PySequence_Size(PyObject* sequence);
PyObject* PySequence_GetItem(PyObject* sequence, Py_ssize_t index);
void Py_DecRef(PyObject* object);
void PyErr_Fetch(PyObject** type, PyObject** value, PyObject** traceback);
void PyErr_NormalizeException(PyObject** type, PyObject** value, PyObject** traceback);
PyObject* PyUnicode_FromString(const char* str);
const char* PyUnicode_AsUTF8(PyObject* object);
PyObject* PyBytes_FromStringAndSize(const char* data, Py_ssize_t size);
int PyBytes_AsStringAndSize(PyObject* object, char** data, Py_ssize_t* size);
void Py_Initialize();
int Py_IsInitialized();
int PyGILState_Ensure();
void PyGILState_Release(int state);
PyThreadState* PyThreadState_New();
void PyEval_RestoreThread(PyThreadState* state);
PyThreadState* PyEval_SaveThread();
PyObject* PyRun_StringFlags(const char* str, int start, PyObject* globals, PyObject* locals, void* flags);
PyObject* PyDict_New();
PyObject* PyObject_GetAttrString(PyObject* object, const char* name);

PyObject* PyObject_CallFunctionObjArgs(PyObject* callable);
PyObject* PyObject_CallFunctionObjArgs(PyObject* callable, PyObject* arg0);
PyObject* PyObject_CallFunctionObjArgs(PyObject* callable, PyObject* arg0, PyObject* arg1);
PyObject* PyObject_CallFunctionObjArgs(PyObject* callable, PyObject* arg0, PyObject* arg1,
                                       PyObject* arg2);
PyObject* PyObject_CallFunctionObjArgs(PyObject* callable, PyObject* arg0, PyObject* arg1,
                                       PyObject* arg2, PyObject* arg3);
PyObject* PyObject_CallFunctionObjArgs(PyObject* callable, PyObject* arg0, PyObject* arg1,
                                       PyObject* arg2, PyObject* arg3, PyObject* arg4);
PyObject* PyObject_CallFunctionObjArgs(PyObject* callable, PyObject* arg0, PyObject* arg1,
                                       PyObject* arg2, PyObject* arg3, PyObject* arg4,
                                       PyObject* arg5);

}  // namespace Jetstream::CPython

#endif  // JETSTREAM_RUNTIME_PYTHON_BRIDGE_CPYTHON_BASE_HH
