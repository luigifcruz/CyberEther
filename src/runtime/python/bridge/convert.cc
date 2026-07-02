#include "runtime/python/bridge/convert.hh"

#include <cmath>
#include <limits>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

#include "jetstream/logger.hh"

namespace Jetstream {

using namespace CPython;

namespace {

template<typename T>
bool FitsSignedInteger(const I64 value) {
    return value >= static_cast<I64>(std::numeric_limits<T>::min()) &&
           value <= static_cast<I64>(std::numeric_limits<T>::max());
}

template<typename T>
bool FitsUnsignedInteger(const U64 value) {
    return value <= static_cast<U64>(std::numeric_limits<T>::max());
}

bool CoerceIntegerToType(const std::type_info& target,
                         const bool isUnsigned,
                         const I64 signedValue,
                         const U64 unsignedValue,
                         std::any& out) {
    if (target == typeid(F32)) {
        out = isUnsigned ? static_cast<F32>(unsignedValue) : static_cast<F32>(signedValue);
        return true;
    }
    if (target == typeid(F64)) {
        out = isUnsigned ? static_cast<F64>(unsignedValue) : static_cast<F64>(signedValue);
        return true;
    }
    if (target == typeid(bool)) {
        out = isUnsigned ? unsignedValue != 0 : signedValue != 0;
        return true;
    }

    if (isUnsigned) {
        if (target == typeid(U64)) {
            out = unsignedValue;
            return true;
        }
        if (target == typeid(U32) && FitsUnsignedInteger<U32>(unsignedValue)) {
            out = static_cast<U32>(unsignedValue);
            return true;
        }
        if (target == typeid(U16) && FitsUnsignedInteger<U16>(unsignedValue)) {
            out = static_cast<U16>(unsignedValue);
            return true;
        }
        if (target == typeid(U8) && FitsUnsignedInteger<U8>(unsignedValue)) {
            out = static_cast<U8>(unsignedValue);
            return true;
        }
        if (target == typeid(I64) &&
            unsignedValue <= static_cast<U64>(std::numeric_limits<I64>::max())) {
            out = static_cast<I64>(unsignedValue);
            return true;
        }
        return false;
    }

    if (target == typeid(I64)) {
        out = signedValue;
        return true;
    }
    if (target == typeid(I32) && FitsSignedInteger<I32>(signedValue)) {
        out = static_cast<I32>(signedValue);
        return true;
    }
    if (target == typeid(I16) && FitsSignedInteger<I16>(signedValue)) {
        out = static_cast<I16>(signedValue);
        return true;
    }
    if (target == typeid(I8) && FitsSignedInteger<I8>(signedValue)) {
        out = static_cast<I8>(signedValue);
        return true;
    }
    if (target == typeid(U64) && signedValue >= 0) {
        out = static_cast<U64>(signedValue);
        return true;
    }
    if (target == typeid(U32) && signedValue >= 0 &&
        FitsUnsignedInteger<U32>(static_cast<U64>(signedValue))) {
        out = static_cast<U32>(signedValue);
        return true;
    }
    if (target == typeid(U16) && signedValue >= 0 &&
        FitsUnsignedInteger<U16>(static_cast<U64>(signedValue))) {
        out = static_cast<U16>(signedValue);
        return true;
    }
    if (target == typeid(U8) && signedValue >= 0 &&
        FitsUnsignedInteger<U8>(static_cast<U64>(signedValue))) {
        out = static_cast<U8>(signedValue);
        return true;
    }

    return false;
}

template<typename T>
bool FitsFloatInteger(const F64 numeric) {
    // The exclusive upper bound (2^digits) is exactly representable in F64,
    // while casting T::max() of 64-bit types rounds up and admits overflow.
    const F64 lower = static_cast<F64>(std::numeric_limits<T>::min());
    const F64 upperExclusive = std::ldexp(1.0, std::numeric_limits<T>::digits);
    return numeric >= lower && numeric < upperExclusive;
}

bool IsIntegerType(const std::type_info& type) {
    return type == typeid(I8) || type == typeid(I16) ||
           type == typeid(I32) || type == typeid(I64) ||
           type == typeid(U8) || type == typeid(U16) ||
           type == typeid(U32) || type == typeid(U64);
}

bool CoerceFloatToType(const std::type_info& target, const F64 numeric, std::any& out) {
    if (target == typeid(F32)) {
        out = static_cast<F32>(numeric);
        return true;
    }
    if (target == typeid(F64)) {
        out = numeric;
        return true;
    }
    if (target == typeid(bool)) {
        out = numeric != 0.0;
        return true;
    }

    if (std::isnan(numeric) || std::floor(numeric) != numeric) {
        return false;
    }

    if (target == typeid(I8) && FitsFloatInteger<I8>(numeric)) {
        out = static_cast<I8>(numeric);
        return true;
    }
    if (target == typeid(I16) && FitsFloatInteger<I16>(numeric)) {
        out = static_cast<I16>(numeric);
        return true;
    }
    if (target == typeid(I32) && FitsFloatInteger<I32>(numeric)) {
        out = static_cast<I32>(numeric);
        return true;
    }
    if (target == typeid(I64) && FitsFloatInteger<I64>(numeric)) {
        out = static_cast<I64>(numeric);
        return true;
    }
    if (target == typeid(U8) && FitsFloatInteger<U8>(numeric)) {
        out = static_cast<U8>(numeric);
        return true;
    }
    if (target == typeid(U16) && FitsFloatInteger<U16>(numeric)) {
        out = static_cast<U16>(numeric);
        return true;
    }
    if (target == typeid(U32) && FitsFloatInteger<U32>(numeric)) {
        out = static_cast<U32>(numeric);
        return true;
    }
    if (target == typeid(U64) && FitsFloatInteger<U64>(numeric)) {
        out = static_cast<U64>(numeric);
        return true;
    }

    return false;
}

Result ConvertPyInteger(PyObject* value, bool& isUnsigned, I64& signedValue, U64& unsignedValue) {
    const auto asSigned = PyLong_AsLongLong(value);
    if (asSigned != -1 || !ClearPythonError()) {
        isUnsigned = false;
        signedValue = asSigned;
        return Result::SUCCESS;
    }

    const auto asUnsigned = PyLong_AsUnsignedLongLong(value);
    if (asUnsigned != static_cast<unsigned long long>(-1) || !ClearPythonError()) {
        isUnsigned = true;
        unsignedValue = asUnsigned;
        return Result::SUCCESS;
    }

    return Result::ERROR;
}

template<typename T, typename Convert>
Result CoerceSequenceToTypedVector(PyObject* value,
                                   const Py_ssize_t size,
                                   Convert&& convert,
                                   std::any& out) {
    std::vector<T> values;
    values.reserve(static_cast<std::size_t>(size));

    for (Py_ssize_t index = 0; index < size; ++index) {
        auto* item = PySequence_GetItem(value, index);
        if (!item) {
            (void)ClearPythonError();
            return Result::ERROR;
        }

        T converted{};
        const auto result = convert(item, converted);
        Py_DecRef(item);
        JST_CHECK(result);

        values.push_back(converted);
    }

    out = std::move(values);
    return Result::SUCCESS;
}

Result ConvertPyFloatItem(PyObject* item, F64& out) {
    out = PyFloat_AsDouble(item);
    if (out == -1.0 && ClearPythonError()) {
        return Result::ERROR;
    }
    return Result::SUCCESS;
}

}  // namespace

namespace {

template<typename T>
bool TypedAnyEquals(const std::any& a, const std::any& b) {
    return std::any_cast<const T&>(a) == std::any_cast<const T&>(b);
}

}  // namespace

bool AnyDeepEquals(const std::any& a, const std::any& b) {
    if (a.has_value() != b.has_value()) {
        return false;
    }
    if (!a.has_value()) {
        return true;
    }
    if (a.type() != b.type()) {
        return false;
    }

    const auto& type = a.type();

    if (type == typeid(F32)) { return TypedAnyEquals<F32>(a, b); }
    if (type == typeid(F64)) { return TypedAnyEquals<F64>(a, b); }
    if (type == typeid(bool)) { return TypedAnyEquals<bool>(a, b); }
    if (type == typeid(I8)) { return TypedAnyEquals<I8>(a, b); }
    if (type == typeid(I16)) { return TypedAnyEquals<I16>(a, b); }
    if (type == typeid(I32)) { return TypedAnyEquals<I32>(a, b); }
    if (type == typeid(I64)) { return TypedAnyEquals<I64>(a, b); }
    if (type == typeid(U8)) { return TypedAnyEquals<U8>(a, b); }
    if (type == typeid(U16)) { return TypedAnyEquals<U16>(a, b); }
    if (type == typeid(U32)) { return TypedAnyEquals<U32>(a, b); }
    if (type == typeid(U64)) { return TypedAnyEquals<U64>(a, b); }
    if (type == typeid(std::string)) { return TypedAnyEquals<std::string>(a, b); }
    if (type == typeid(std::vector<F32>)) { return TypedAnyEquals<std::vector<F32>>(a, b); }
    if (type == typeid(std::vector<F64>)) { return TypedAnyEquals<std::vector<F64>>(a, b); }
    if (type == typeid(std::vector<U64>)) { return TypedAnyEquals<std::vector<U64>>(a, b); }

    if (type == typeid(Parser::Map)) {
        const auto& mapA = std::any_cast<const Parser::Map&>(a);
        const auto& mapB = std::any_cast<const Parser::Map&>(b);
        if (mapA.size() != mapB.size()) {
            return false;
        }
        for (const auto& [key, entry] : mapA) {
            if (!mapB.contains(key) || !AnyDeepEquals(entry, mapB.at(key))) {
                return false;
            }
        }
        return true;
    }

    if (type == typeid(Parser::Sequence)) {
        const auto& sequenceA = std::any_cast<const Parser::Sequence&>(a);
        const auto& sequenceB = std::any_cast<const Parser::Sequence&>(b);
        if (sequenceA.size() != sequenceB.size()) {
            return false;
        }
        for (std::size_t index = 0; index < sequenceA.size(); ++index) {
            if (!AnyDeepEquals(sequenceA[index], sequenceB[index])) {
                return false;
            }
        }
        return true;
    }

    return false;
}

bool ClearPythonError() {
    PyObject* type = nullptr;
    PyObject* value = nullptr;
    PyObject* traceback = nullptr;
    PyErr_Fetch(&type, &value, &traceback);

    if (!type && !value && !traceback) {
        return false;
    }

    if (type) { Py_DecRef(type); }
    if (value) { Py_DecRef(value); }
    if (traceback) { Py_DecRef(traceback); }
    return true;
}

Result ClassifyPyObject(PyObject* classify, PyObject* value, I64& code) {
    auto* codeObject = PyObject_CallFunctionObjArgs(classify, value);
    if (!codeObject) {
        (void)ClearPythonError();
        return Result::ERROR;
    }

    code = PyLong_AsLongLong(codeObject);
    Py_DecRef(codeObject);
    if (code == -1 && ClearPythonError()) {
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

PyObject* AnyToPyObject(const std::any& value) {
    if (value.type() == typeid(F32)) {
        return PyFloat_FromDouble(static_cast<double>(std::any_cast<F32>(value)));
    }
    if (value.type() == typeid(F64)) {
        return PyFloat_FromDouble(std::any_cast<F64>(value));
    }
    if (value.type() == typeid(bool)) {
        return PyBool_FromLong(std::any_cast<bool>(value) ? 1 : 0);
    }
    if (value.type() == typeid(I8)) {
        return PyLong_FromLongLong(std::any_cast<I8>(value));
    }
    if (value.type() == typeid(I16)) {
        return PyLong_FromLongLong(std::any_cast<I16>(value));
    }
    if (value.type() == typeid(I32)) {
        return PyLong_FromLongLong(std::any_cast<I32>(value));
    }
    if (value.type() == typeid(I64)) {
        return PyLong_FromLongLong(std::any_cast<I64>(value));
    }
    if (value.type() == typeid(U8)) {
        return PyLong_FromUnsignedLongLong(std::any_cast<U8>(value));
    }
    if (value.type() == typeid(U16)) {
        return PyLong_FromUnsignedLongLong(std::any_cast<U16>(value));
    }
    if (value.type() == typeid(U32)) {
        return PyLong_FromUnsignedLongLong(std::any_cast<U32>(value));
    }
    if (value.type() == typeid(U64)) {
        return PyLong_FromUnsignedLongLong(std::any_cast<U64>(value));
    }
    if (value.type() == typeid(std::string)) {
        return PyUnicode_FromString(std::any_cast<const std::string&>(value).c_str());
    }
    if (value.type() == typeid(std::pair<std::string, F32>)) {
        const auto& pair = std::any_cast<const std::pair<std::string, F32>&>(value);
        auto* label = PyUnicode_FromString(pair.first.c_str());
        auto* fraction = PyFloat_FromDouble(static_cast<double>(pair.second));
        auto* tuple = PyTuple_New(2);
        if (!label || !fraction || !tuple) {
            if (label) { Py_DecRef(label); }
            if (fraction) { Py_DecRef(fraction); }
            if (tuple) { Py_DecRef(tuple); }
            return nullptr;
        }
        if (PyTuple_SetItem(tuple, 0, label) != 0) {
            Py_DecRef(fraction);
            Py_DecRef(tuple);
            return nullptr;
        }
        if (PyTuple_SetItem(tuple, 1, fraction) != 0) {
            Py_DecRef(tuple);
            return nullptr;
        }
        return tuple;
    }
    if (value.type() == typeid(std::vector<F32>)) {
        const auto& values = std::any_cast<const std::vector<F32>&>(value);
        auto* tuple = PyTuple_New(static_cast<Py_ssize_t>(values.size()));
        if (!tuple) {
            return nullptr;
        }
        for (std::size_t index = 0; index < values.size(); ++index) {
            auto* object = PyFloat_FromDouble(static_cast<double>(values[index]));
            if (!object || PyTuple_SetItem(tuple, static_cast<Py_ssize_t>(index), object) != 0) {
                if (!object) { Py_DecRef(tuple); return nullptr; }
                Py_DecRef(tuple);
                return nullptr;
            }
        }
        return tuple;
    }
    if (value.type() == typeid(std::vector<F64>)) {
        const auto& values = std::any_cast<const std::vector<F64>&>(value);
        auto* tuple = PyTuple_New(static_cast<Py_ssize_t>(values.size()));
        if (!tuple) {
            return nullptr;
        }
        for (std::size_t index = 0; index < values.size(); ++index) {
            auto* object = PyFloat_FromDouble(values[index]);
            if (!object || PyTuple_SetItem(tuple, static_cast<Py_ssize_t>(index), object) != 0) {
                if (!object) { Py_DecRef(tuple); return nullptr; }
                Py_DecRef(tuple);
                return nullptr;
            }
        }
        return tuple;
    }
    if (value.type() == typeid(std::vector<U64>)) {
        const auto& values = std::any_cast<const std::vector<U64>&>(value);
        auto* tuple = PyTuple_New(static_cast<Py_ssize_t>(values.size()));
        if (!tuple) {
            return nullptr;
        }
        for (std::size_t index = 0; index < values.size(); ++index) {
            auto* object = PyLong_FromUnsignedLongLong(values[index]);
            if (!object || PyTuple_SetItem(tuple, static_cast<Py_ssize_t>(index), object) != 0) {
                if (!object) { Py_DecRef(tuple); return nullptr; }
                Py_DecRef(tuple);
                return nullptr;
            }
        }
        return tuple;
    }
    if (value.type() == typeid(Parser::Map)) {
        const auto& map = std::any_cast<const Parser::Map&>(value);
        auto* dict = PyDict_New();
        if (!dict) {
            return nullptr;
        }

        for (const auto& [key, entry] : map) {
            auto* object = AnyToPyObject(entry);
            if (!object) {
                JST_TRACE("[RUNTIME_CONTEXT_PYTHON] Skipping unsupported value for key '{}'.", key);
                (void)ClearPythonError();
                continue;
            }

            if (PyDict_SetItemString(dict, key.c_str(), object) != 0) {
                Py_DecRef(object);
                Py_DecRef(dict);
                return nullptr;
            }
            Py_DecRef(object);
        }

        return dict;
    }
    if (value.type() == typeid(Parser::Sequence)) {
        const auto& sequence = std::any_cast<const Parser::Sequence&>(value);
        auto* tuple = PyTuple_New(static_cast<Py_ssize_t>(sequence.size()));
        if (!tuple) {
            return nullptr;
        }

        for (std::size_t index = 0; index < sequence.size(); ++index) {
            auto* object = AnyToPyObject(sequence[index]);
            if (!object) {
                Py_DecRef(tuple);
                return nullptr;
            }

            if (PyTuple_SetItem(tuple, static_cast<Py_ssize_t>(index), object) != 0) {
                Py_DecRef(tuple);
                return nullptr;
            }
        }

        return tuple;
    }
    return nullptr;
}

Result PyObjectToAny(PyObject* classify,
                     PyObject* value,
                     const std::any& existing,
                     std::any& out) {
    I64 code = -1;
    JST_CHECK(ClassifyPyObject(classify, value, code));

    if (code == 3) {
        const char* str = PyUnicode_AsUTF8(value);
        if (!str) {
            (void)ClearPythonError();
            return Result::ERROR;
        }
        out = std::string(str);
        return Result::SUCCESS;
    }

    if (code == 4) {
        const auto* existingMap = std::any_cast<Parser::Map>(&existing);

        Parser::Map map;
        PyObject* entryKey = nullptr;
        PyObject* entryValue = nullptr;
        Py_ssize_t position = 0;

        while (PyDict_Next(value, &position, &entryKey, &entryValue)) {
            const char* keyStr = PyUnicode_AsUTF8(entryKey);
            if (!keyStr) {
                (void)ClearPythonError();
                return Result::ERROR;
            }

            std::any entryExisting;
            if (existingMap && existingMap->contains(keyStr)) {
                entryExisting = existingMap->at(keyStr);
            }

            std::any converted;
            JST_CHECK(PyObjectToAny(classify, entryValue, entryExisting, converted));
            map[keyStr] = std::move(converted);
        }

        out = std::move(map);
        return Result::SUCCESS;
    }

    if (code == 5) {
        const auto size = PySequence_Size(value);
        if (size < 0) {
            (void)ClearPythonError();
            return Result::ERROR;
        }

        if (existing.has_value()) {
            if (existing.type() == typeid(std::vector<F32>)) {
                return CoerceSequenceToTypedVector<F32>(value, size,
                    [](PyObject* item, F32& converted) {
                        F64 numeric = 0.0;
                        JST_CHECK(ConvertPyFloatItem(item, numeric));
                        converted = static_cast<F32>(numeric);
                        return Result::SUCCESS;
                    }, out);
            }
            if (existing.type() == typeid(std::vector<F64>)) {
                return CoerceSequenceToTypedVector<F64>(value, size,
                    [](PyObject* item, F64& converted) {
                        return ConvertPyFloatItem(item, converted);
                    }, out);
            }
            if (existing.type() == typeid(std::vector<U64>)) {
                return CoerceSequenceToTypedVector<U64>(value, size,
                    [](PyObject* item, U64& converted) {
                        const auto numeric = PyLong_AsUnsignedLongLong(item);
                        if (numeric == static_cast<unsigned long long>(-1) && ClearPythonError()) {
                            return Result::ERROR;
                        }
                        converted = numeric;
                        return Result::SUCCESS;
                    }, out);
            }
        }

        const auto* existingSequence = std::any_cast<Parser::Sequence>(&existing);

        if (!existing.has_value() && size > 0) {
            bool allFloats = true;
            bool allIntegers = true;
            for (Py_ssize_t index = 0; index < size && (allFloats || allIntegers); ++index) {
                auto* item = PySequence_GetItem(value, index);
                if (!item) {
                    (void)ClearPythonError();
                    return Result::ERROR;
                }

                I64 itemCode = -1;
                const auto result = ClassifyPyObject(classify, item, itemCode);
                Py_DecRef(item);
                JST_CHECK(result);

                allFloats = allFloats && itemCode == 2;
                allIntegers = allIntegers && itemCode == 1;
            }

            if (allFloats) {
                return CoerceSequenceToTypedVector<F64>(value, size,
                    [](PyObject* item, F64& converted) {
                        return ConvertPyFloatItem(item, converted);
                    }, out);
            }

            if (allIntegers) {
                std::any typedVector;
                const auto result = CoerceSequenceToTypedVector<U64>(value, size,
                    [](PyObject* item, U64& converted) {
                        const auto numeric = PyLong_AsUnsignedLongLong(item);
                        if (numeric == static_cast<unsigned long long>(-1) && ClearPythonError()) {
                            return Result::ERROR;
                        }
                        converted = numeric;
                        return Result::SUCCESS;
                    }, typedVector);
                if (result == Result::SUCCESS) {
                    out = std::move(typedVector);
                    return Result::SUCCESS;
                }
            }
        }

        Parser::Sequence sequence;
        sequence.reserve(static_cast<std::size_t>(size));

        for (Py_ssize_t index = 0; index < size; ++index) {
            auto* item = PySequence_GetItem(value, index);
            if (!item) {
                (void)ClearPythonError();
                return Result::ERROR;
            }

            std::any entryExisting;
            if (existingSequence && static_cast<std::size_t>(index) < existingSequence->size()) {
                entryExisting = existingSequence->at(static_cast<std::size_t>(index));
            }

            std::any converted;
            const auto result = PyObjectToAny(classify, item, entryExisting, converted);
            Py_DecRef(item);
            JST_CHECK(result);

            sequence.push_back(std::move(converted));
        }

        out = std::move(sequence);
        return Result::SUCCESS;
    }

    if (code == 0) {
        const bool flag = PyLong_AsLongLong(value) != 0;
        if (existing.has_value() &&
            CoerceIntegerToType(existing.type(), false, flag ? 1 : 0, 0, out)) {
            return Result::SUCCESS;
        }
        out = flag;
        return Result::SUCCESS;
    }

    if (code == 1) {
        bool isUnsigned = false;
        I64 signedValue = 0;
        U64 unsignedValue = 0;
        if (ConvertPyInteger(value, isUnsigned, signedValue, unsignedValue) != Result::SUCCESS) {
            JST_WARN("[RUNTIME_CONTEXT_PYTHON] Integer value does not fit in 64 bits.");
            return Result::ERROR;
        }

        if (existing.has_value() &&
            CoerceIntegerToType(existing.type(), isUnsigned, signedValue, unsignedValue, out)) {
            return Result::SUCCESS;
        }

        if (isUnsigned) {
            out = unsignedValue;
        } else {
            out = signedValue;
        }
        return Result::SUCCESS;
    }

    if (code == 2) {
        const F64 numeric = PyFloat_AsDouble(value);
        if (numeric == -1.0 && ClearPythonError()) {
            return Result::ERROR;
        }

        if (existing.has_value()) {
            if (CoerceFloatToType(existing.type(), numeric, out)) {
                return Result::SUCCESS;
            }

            if (IsIntegerType(existing.type())) {
                JST_WARN("[RUNTIME_CONTEXT_PYTHON] Rejecting non-integral or out-of-range "
                         "float assigned to an integer value.");
                return Result::ERROR;
            }
        }

        out = numeric;
        return Result::SUCCESS;
    }

    return Result::ERROR;
}

}  // namespace Jetstream
