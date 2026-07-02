#ifndef JETSTREAM_RUNTIME_PYTHON_BRIDGE_CONVERT_HH
#define JETSTREAM_RUNTIME_PYTHON_BRIDGE_CONVERT_HH

#include <any>

#include "runtime/python/bridge/cpython/base.hh"

#include "jetstream/parser.hh"

namespace Jetstream {

bool ClearPythonError();

bool AnyDeepEquals(const std::any& a, const std::any& b);

Result ClassifyPyObject(CPython::PyObject* classify,
                        CPython::PyObject* value,
                        I64& code);

CPython::PyObject* AnyToPyObject(const std::any& value, CPython::PyObject* converters = nullptr);

Result PyObjectToAny(CPython::PyObject* classify,
                     CPython::PyObject* value,
                     const std::any& existing,
                     std::any& out);

}  // namespace Jetstream

#endif  // JETSTREAM_RUNTIME_PYTHON_BRIDGE_CONVERT_HH
