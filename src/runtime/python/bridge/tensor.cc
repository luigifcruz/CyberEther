#include <cstddef>
#include <limits>
#include <string>

#include "runtime/python/bridge/base.hh"
#include "runtime/python/bridge/cpython/base.hh"

#include "jetstream/logger.hh"

namespace Jetstream {

using namespace CPython;

namespace {

inline constexpr int kPyBufferRead = 0x100;
inline constexpr int kPyBufferWrite = 0x200;

const char* NumpyDtypeName(const DataType dtype) {
    switch (dtype) {
        case DataType::I8: return "int8";
        case DataType::I16: return "int16";
        case DataType::I32: return "int32";
        case DataType::I64: return "int64";
        case DataType::U8: return "uint8";
        case DataType::U16: return "uint16";
        case DataType::U32: return "uint32";
        case DataType::U64: return "uint64";
        case DataType::F32: return "float32";
        case DataType::F64: return "float64";
        case DataType::CF32: return "complex64";
        case DataType::CF64: return "complex128";
        default: return nullptr;
    }
}

Result ValidateTensorForNumpy(const Tensor& tensor, const std::string& name, const bool writable) {
    const auto role = writable ? "output" : "input";

    if (tensor.device() != DeviceType::CPU) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Python runtime currently supports CPU {} tensors only.", role);
        return Result::ERROR;
    }

    if (!tensor.contiguous()) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Python runtime currently requires contiguous {} tensors.", role);
        return Result::ERROR;
    }

    if (!NumpyDtypeName(tensor.dtype())) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Tensor '{}' dtype '{}' cannot be exposed to NumPy yet.",
                  name, tensor.dtype());
        return Result::ERROR;
    }

    if (tensor.sizeBytes() > static_cast<U64>(std::numeric_limits<Py_ssize_t>::max())) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Tensor '{}' is too large to expose to Python.", name);
        return Result::ERROR;
    }

    if (tensor.sizeBytes() > 0 && !tensor.data()) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Tensor '{}' data is null.", name);
        return Result::ERROR;
    }

    for (Index axis = 0; axis < tensor.rank(); ++axis) {
        if (tensor.shape(axis) > static_cast<U64>(std::numeric_limits<long long>::max())) {
            JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Tensor '{}' shape exceeds Python integer limits.", name);
            return Result::ERROR;
        }
    }

    return Result::SUCCESS;
}

PyObject* CreateShapeTuple(const Tensor& tensor) {
    if (tensor.rank() > static_cast<U64>(std::numeric_limits<Py_ssize_t>::max())) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Tensor rank exceeds Python tuple limits.");
        return nullptr;
    }

    auto* shape = PyTuple_New(static_cast<Py_ssize_t>(tensor.rank()));
    if (!shape) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Can't create Python shape tuple.");
        return nullptr;
    }

    for (Index axis = 0; axis < tensor.rank(); ++axis) {
        auto* dimension = PyLong_FromLongLong(static_cast<long long>(tensor.shape(axis)));
        if (!dimension) {
            JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Can't create Python shape dimension.");
            Py_DecRef(shape);
            return nullptr;
        }

        if (PyTuple_SetItem(shape, static_cast<Py_ssize_t>(axis), dimension) != 0) {
            JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Can't populate Python shape tuple.");
            Py_DecRef(dimension);
            Py_DecRef(shape);
            return nullptr;
        }
    }

    return shape;
}

PyObject* CreateTensorSpec(const Tensor& tensor,
                           const std::string& name,
                           const bool writable) {
    if (ValidateTensorForNumpy(tensor, name, writable) != Result::SUCCESS) {
        return nullptr;
    }

    auto* data = const_cast<char*>(static_cast<const char*>(tensor.data()));
    auto* memory = PyMemoryView_FromMemory(data,
                                           static_cast<Py_ssize_t>(tensor.sizeBytes()),
                                           writable ? kPyBufferWrite : kPyBufferRead);
    auto* dtype = PyUnicode_FromString(NumpyDtypeName(tensor.dtype()));
    auto* shape = CreateShapeTuple(tensor);
    auto* spec = PyTuple_New(3);

    auto cleanup = [&]() {
        if (memory) { Py_DecRef(memory); }
        if (dtype) { Py_DecRef(dtype); }
        if (shape) { Py_DecRef(shape); }
        if (spec) { Py_DecRef(spec); }
    };

    auto setTupleItem = [&](Py_ssize_t index, auto*& item) {
        if (PyTuple_SetItem(spec, index, item) != 0) {
            JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Can't prepare Python tensor spec for tensor '{}'.", name);
            return false;
        }
        item = nullptr;
        return true;
    };

    if (!memory || !dtype || !shape || !spec) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Can't prepare Python tensor spec for tensor '{}'.", name);
        cleanup();
        return nullptr;
    }

    if (!setTupleItem(0, memory) || !setTupleItem(1, dtype) || !setTupleItem(2, shape)) {
        cleanup();
        return nullptr;
    }

    return spec;
}

PyObject* CreateTensorSpecs(const Module::Interface::EntryList& order,
                            const TensorMap& tensors,
                            const bool writable) {
    if (order.size() > static_cast<std::size_t>(std::numeric_limits<Py_ssize_t>::max())) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Tensor context exceeds Python tuple limits.");
        return nullptr;
    }

    auto* specs = PyTuple_New(static_cast<Py_ssize_t>(order.size()));
    if (!specs) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Can't create Python tensor spec tuple.");
        return nullptr;
    }

    for (std::size_t index = 0; index < order.size(); ++index) {
        const auto& name = order[index];
        if (!tensors.contains(name)) {
            JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Missing {} tensor '{}'.",
                      writable ? "output" : "input", name);
            Py_DecRef(specs);
            return nullptr;
        }

        auto* spec = CreateTensorSpec(tensors.at(name).tensor, name, writable);
        if (!spec) {
            Py_DecRef(specs);
            return nullptr;
        }

        if (PyTuple_SetItem(specs, static_cast<Py_ssize_t>(index), spec) != 0) {
            JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Can't add tensor '{}' to Python spec index {}.",
                      name, index);
            Py_DecRef(spec);
            Py_DecRef(specs);
            return nullptr;
        }
    }

    return specs;
}

}  // namespace

PyObject* Bridge::createTensorContext(const Module::Interface::EntryList& inputOrder,
                                      const TensorMap& inputs,
                                      const Module::Interface::EntryList& outputOrder,
                                      const TensorMap& outputs) {
    if (!globals) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Can't create Python tensor context without globals.");
        return nullptr;
    }

    auto* inputSpecs = CreateTensorSpecs(inputOrder, inputs, false);
    if (!inputSpecs) {
        return nullptr;
    }

    auto* outputSpecs = CreateTensorSpecs(outputOrder, outputs, true);
    if (!outputSpecs) {
        Py_DecRef(inputSpecs);
        return nullptr;
    }

    auto* createBridge = PyDict_GetItemString(globals, "_jetstream_create_bridge");
    if (!createBridge || !PyCallable_Check(createBridge)) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Python runtime helper '_jetstream_create_bridge' is unavailable.");
        Py_DecRef(outputSpecs);
        Py_DecRef(inputSpecs);
        return nullptr;
    }

    auto* tensorContext = PyObject_CallFunctionObjArgs(createBridge, inputSpecs, outputSpecs);
    Py_DecRef(outputSpecs);
    Py_DecRef(inputSpecs);
    if (!tensorContext) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Can't create Python tensor context.");
        return nullptr;
    }

    return tensorContext;
}

}  // namespace Jetstream
