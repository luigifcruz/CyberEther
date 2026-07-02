#include <any>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

#include "runtime/python/bridge/base.hh"
#include "runtime/python/bridge/convert.hh"
#include "runtime/python/bridge/cpython/base.hh"

#include "jetstream/logger.hh"

namespace Jetstream {

using namespace CPython;

namespace {

inline constexpr int kPyBufferRead = 0x100;
inline constexpr int kPyBufferWrite = 0x200;

const char* PythonDeviceName(const DeviceType device) {
    switch (device) {
        case DeviceType::CPU: return "cpu";
        case DeviceType::CUDA: return "cuda";
        default: return nullptr;
    }
}

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

U64 TensorSpanBytes(const Tensor& tensor) {
    if (tensor.size() == 0) {
        return 0;
    }

    U64 lastElement = 0;
    for (const auto& backstride : tensor.backstride()) {
        lastElement += backstride;
    }

    return (lastElement + 1) * tensor.elementSize();
}

Result ValidateTensorForNumpy(const Tensor& tensor, const std::string& name, const bool writable) {
    const auto role = writable ? "output" : "input";

    if (!PythonDeviceName(tensor.device())) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Python runtime currently supports CPU and CUDA {} tensors only.", role);
        return Result::ERROR;
    }

    if (!NumpyDtypeName(tensor.dtype())) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Tensor '{}' dtype '{}' cannot be exposed to NumPy yet.",
                  name, tensor.dtype());
        return Result::ERROR;
    }

    const U64 spanBytes = TensorSpanBytes(tensor);
    if (spanBytes > static_cast<U64>(std::numeric_limits<Py_ssize_t>::max())) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Tensor '{}' is too large to expose to Python.", name);
        return Result::ERROR;
    }

    if (spanBytes > 0 && !tensor.data()) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Tensor '{}' data is null.", name);
        return Result::ERROR;
    }

    for (Index axis = 0; axis < tensor.rank(); ++axis) {
        if (tensor.shape(axis) > static_cast<U64>(std::numeric_limits<long long>::max())) {
            JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Tensor '{}' shape exceeds Python integer limits.", name);
            return Result::ERROR;
        }

        if (tensor.stride(axis) > static_cast<U64>(std::numeric_limits<long long>::max()) / tensor.elementSize()) {
            JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Tensor '{}' stride exceeds Python integer limits.", name);
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

PyObject* CreateStridesTuple(const Tensor& tensor) {
    if (tensor.rank() > static_cast<U64>(std::numeric_limits<Py_ssize_t>::max())) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Tensor rank exceeds Python tuple limits.");
        return nullptr;
    }

    auto* strides = PyTuple_New(static_cast<Py_ssize_t>(tensor.rank()));
    if (!strides) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Can't create Python strides tuple.");
        return nullptr;
    }

    for (Index axis = 0; axis < tensor.rank(); ++axis) {
        const U64 strideBytes = tensor.stride(axis) * tensor.elementSize();
        auto* stride = PyLong_FromLongLong(static_cast<long long>(strideBytes));
        if (!stride) {
            JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Can't create Python stride dimension.");
            Py_DecRef(strides);
            return nullptr;
        }

        if (PyTuple_SetItem(strides, static_cast<Py_ssize_t>(axis), stride) != 0) {
            JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Can't populate Python strides tuple.");
            Py_DecRef(stride);
            Py_DecRef(strides);
            return nullptr;
        }
    }

    return strides;
}

PyObject* CreateMemoryObject(const Tensor& tensor, const bool writable) {
    const U64 spanBytes = TensorSpanBytes(tensor);

    if (tensor.device() == DeviceType::CPU) {
        auto* data = const_cast<char*>(static_cast<const char*>(tensor.data()));
        return PyMemoryView_FromMemory(data,
                                       static_cast<Py_ssize_t>(spanBytes),
                                       writable ? kPyBufferWrite : kPyBufferRead);
    }

    const auto pointer = reinterpret_cast<std::uintptr_t>(tensor.data()) + tensor.offsetBytes();
    auto* pointerObject = PyLong_FromUnsignedLongLong(static_cast<unsigned long long>(pointer));
    auto* spanObject = PyLong_FromUnsignedLongLong(static_cast<unsigned long long>(spanBytes));
    auto* memory = PyTuple_New(2);
    if (!pointerObject || !spanObject || !memory) {
        if (pointerObject) { Py_DecRef(pointerObject); }
        if (spanObject) { Py_DecRef(spanObject); }
        if (memory) { Py_DecRef(memory); }
        return nullptr;
    }

    if (PyTuple_SetItem(memory, 0, pointerObject) != 0) {
        Py_DecRef(pointerObject);
        Py_DecRef(spanObject);
        Py_DecRef(memory);
        return nullptr;
    }

    if (PyTuple_SetItem(memory, 1, spanObject) != 0) {
        Py_DecRef(spanObject);
        Py_DecRef(memory);
        return nullptr;
    }

    return memory;
}

PyObject* CreateTensorSpec(const Tensor& tensor,
                           const std::string& name,
                           const bool writable) {
    if (ValidateTensorForNumpy(tensor, name, writable) != Result::SUCCESS) {
        return nullptr;
    }

    auto* device = PyUnicode_FromString(PythonDeviceName(tensor.device()));
    auto* memory = CreateMemoryObject(tensor, writable);
    auto* dtype = PyUnicode_FromString(NumpyDtypeName(tensor.dtype()));
    auto* shape = CreateShapeTuple(tensor);
    auto* strides = CreateStridesTuple(tensor);
    auto* writableFlag = PyBool_FromLong(writable ? 1 : 0);
    auto* spec = PyTuple_New(6);

    auto cleanup = [&]() {
        if (device) { Py_DecRef(device); }
        if (memory) { Py_DecRef(memory); }
        if (dtype) { Py_DecRef(dtype); }
        if (shape) { Py_DecRef(shape); }
        if (strides) { Py_DecRef(strides); }
        if (writableFlag) { Py_DecRef(writableFlag); }
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

    if (!device || !memory || !dtype || !shape || !strides || !writableFlag || !spec) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Can't prepare Python tensor spec for tensor '{}'.", name);
        cleanup();
        return nullptr;
    }

    if (!setTupleItem(0, device) ||
        !setTupleItem(1, memory) ||
        !setTupleItem(2, dtype) ||
        !setTupleItem(3, shape) ||
        !setTupleItem(4, strides) ||
        !setTupleItem(5, writableFlag)) {
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

    auto* inputAttributes = createAttributeDicts(inputOrder, inputs, inputAttributePorts);
    if (!inputAttributes) {
        Py_DecRef(outputSpecs);
        Py_DecRef(inputSpecs);
        destroyAttributePorts();
        return nullptr;
    }

    auto* outputAttributes = createAttributeDicts(outputOrder, outputs, outputAttributePorts);
    if (!outputAttributes) {
        Py_DecRef(inputAttributes);
        Py_DecRef(outputSpecs);
        Py_DecRef(inputSpecs);
        destroyAttributePorts();
        return nullptr;
    }

    auto* environment = createEnvironmentDict();
    if (!environment) {
        Py_DecRef(outputAttributes);
        Py_DecRef(inputAttributes);
        Py_DecRef(outputSpecs);
        Py_DecRef(inputSpecs);
        destroyAttributePorts();
        return nullptr;
    }

    auto* createBridge = PyDict_GetItemString(globals, "_jetstream_create_bridge");
    if (!createBridge || !PyCallable_Check(createBridge)) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Python runtime helper '_jetstream_create_bridge' is unavailable.");
        Py_DecRef(environment);
        Py_DecRef(outputAttributes);
        Py_DecRef(inputAttributes);
        Py_DecRef(outputSpecs);
        Py_DecRef(inputSpecs);
        destroyAttributePorts();
        destroyEnvironmentDict();
        return nullptr;
    }

    auto* tensorContext = PyObject_CallFunctionObjArgs(createBridge,
                                                       inputSpecs,
                                                       outputSpecs,
                                                       inputAttributes,
                                                       outputAttributes,
                                                       environment);
    Py_DecRef(environment);
    Py_DecRef(outputAttributes);
    Py_DecRef(inputAttributes);
    Py_DecRef(outputSpecs);
    Py_DecRef(inputSpecs);
    if (!tensorContext) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Can't create Python tensor context.");
        destroyAttributePorts();
        destroyEnvironmentDict();
        return nullptr;
    }

    return tensorContext;
}

PyObject* Bridge::createAttributeDicts(const Module::Interface::EntryList& order,
                                       const TensorMap& tensors,
                                       std::vector<AttributePort>& ports) {
    auto* dicts = PyTuple_New(static_cast<Py_ssize_t>(order.size()));
    if (!dicts) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Can't create Python attribute dict tuple.");
        return nullptr;
    }

    for (std::size_t index = 0; index < order.size(); ++index) {
        const auto& name = order[index];
        if (!tensors.contains(name)) {
            JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Missing tensor '{}' for attribute dict.", name);
            Py_DecRef(dicts);
            return nullptr;
        }

        auto* dict = PyDict_New();
        if (!dict) {
            JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Can't create Python attribute dict.");
            Py_DecRef(dicts);
            return nullptr;
        }

        Py_IncRef(dict);
        if (PyTuple_SetItem(dicts, static_cast<Py_ssize_t>(index), dict) != 0) {
            JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Can't populate Python attribute dict tuple.");
            Py_DecRef(dict);
            Py_DecRef(dicts);
            return nullptr;
        }

        ports.push_back({tensors.at(name).tensor, dict, {}});
    }

    return dicts;
}

void Bridge::refreshAttributes() {
    auto refreshPort = [](AttributePort& port, const bool keepSnapshot) {
        if (keepSnapshot) {
            for (auto& [_, object] : port.snapshot) {
                Py_DecRef(object);
            }
            port.snapshot.clear();
        }

        PyDict_Clear(port.dict);

        for (const auto& key : port.tensor.attributeKeys()) {
            auto* object = AnyToPyObject(port.tensor.attribute(key));
            if (!object) {
                JST_TRACE("[RUNTIME_CONTEXT_PYTHON] Skipping unsupported attribute '{}'.", key);
                (void)ClearPythonError();
                continue;
            }

            if (PyDict_SetItemString(port.dict, key.c_str(), object) != 0) {
                Py_DecRef(object);
                (void)ClearPythonError();
                continue;
            }

            if (keepSnapshot) {
                port.snapshot[key] = object;
            } else {
                Py_DecRef(object);
            }
        }
    };

    for (auto& port : inputAttributePorts) {
        refreshPort(port, false);
    }
    for (auto& port : outputAttributePorts) {
        refreshPort(port, true);
    }
}

void Bridge::flushAttributes() {
    if (outputAttributePorts.empty() || !globals) {
        return;
    }

    auto* classify = PyDict_GetItemString(globals, "_jetstream_classify_attribute");
    if (!classify || !PyCallable_Check(classify)) {
        JST_ERROR("[RUNTIME_CONTEXT_PYTHON] Python runtime helper '_jetstream_classify_attribute' is unavailable.");
        return;
    }

    for (auto& port : outputAttributePorts) {
        PyObject* key = nullptr;
        PyObject* value = nullptr;
        Py_ssize_t position = 0;

        while (PyDict_Next(port.dict, &position, &key, &value)) {
            const char* keyStr = PyUnicode_AsUTF8(key);
            if (!keyStr) {
                (void)ClearPythonError();
                continue;
            }

            const auto it = port.snapshot.find(keyStr);
            const bool sameObject = it != port.snapshot.end() && it->second == value;

            if (sameObject) {
                I64 code = -1;
                if (ClassifyPyObject(classify, value, code) != Result::SUCCESS) {
                    continue;
                }
                if (code != 4 && code != 5) {
                    continue;
                }
            }

            std::any existing;
            if (port.tensor.hasAttribute(keyStr)) {
                existing = port.tensor.attribute(keyStr);
            }

            std::any converted;
            if (PyObjectToAny(classify, value, existing, converted) != Result::SUCCESS) {
                JST_WARN("[RUNTIME_CONTEXT_PYTHON] Ignoring unsupported attribute '{}' value.", keyStr);
                continue;
            }

            if (sameObject && AnyDeepEquals(converted, existing)) {
                continue;
            }

            (void)port.tensor.setAttribute(keyStr, converted);
        }
    }
}

void Bridge::destroyAttributePorts() {
    auto destroyPorts = [](std::vector<AttributePort>& ports) {
        for (auto& port : ports) {
            for (auto& [_, object] : port.snapshot) {
                Py_DecRef(object);
            }
            port.snapshot.clear();

            if (port.dict) {
                Py_DecRef(port.dict);
                port.dict = nullptr;
            }
        }
        ports.clear();
    };

    destroyPorts(inputAttributePorts);
    destroyPorts(outputAttributePorts);
}

}  // namespace Jetstream
