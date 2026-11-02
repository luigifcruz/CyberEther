#include "jetstream/memory/attributes.hh"

#include <cmath>
#include <string>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

#include "jetstream/logger.hh"
#include "jetstream/macros.hh"
#include "jetstream/memory/tensor.hh"

namespace Jetstream {

namespace {

const char* AttributeTypeName(const std::type_info& type) {
    static const std::unordered_map<std::type_index, const char*> names = {
        {typeid(bool), "bool"},
        {typeid(std::string), "string"},
        {typeid(I8), "I8"},
        {typeid(I16), "I16"},
        {typeid(I32), "I32"},
        {typeid(I64), "I64"},
        {typeid(U8), "U8"},
        {typeid(U16), "U16"},
        {typeid(U32), "U32"},
        {typeid(U64), "U64"},
        {typeid(F32), "F32"},
        {typeid(F64), "F64"},
        {typeid(CF32), "CF32"},
        {typeid(CF64), "CF64"},
        {typeid(std::vector<I64>), "vector<I64>"},
        {typeid(std::vector<U64>), "vector<U64>"},
        {typeid(std::vector<F32>), "vector<F32>"},
        {typeid(std::vector<F64>), "vector<F64>"},
    };

    const auto it = names.find(type);
    return it != names.end() ? it->second : type.name();
}

Result CheckFiniteF32(const std::string_view key,
                      const std::string_view expected,
                      const std::any& value) {
    const auto* scalar = std::any_cast<F32>(&value);
    if (!scalar) {
        return AttributeTypeError(key, expected, value);
    }
    if (!std::isfinite(*scalar)) {
        JST_ERROR("[MEMORY:TENSOR] Attribute '{}' must be finite (got {}).", key, *scalar);
        return Result::ERROR;
    }
    return Result::SUCCESS;
}

template<typename T>
Result CheckVector(const std::string_view key,
                   const std::string_view expected,
                   const std::any& value) {
    const auto* vector = std::any_cast<std::vector<T>>(&value);
    if (!vector) {
        return AttributeTypeError(key, expected, value);
    }
    if (vector->empty()) {
        JST_ERROR("[MEMORY:TENSOR] Attribute '{}' cannot be empty.", key);
        return Result::ERROR;
    }
    if constexpr (std::is_floating_point_v<T>) {
        for (U64 i = 0; i < vector->size(); ++i) {
            if (!std::isfinite((*vector)[i])) {
                JST_ERROR("[MEMORY:TENSOR] Attribute '{}' element #{} must be finite (got {}).",
                          key, i, (*vector)[i]);
                return Result::ERROR;
            }
        }
    }
    return Result::SUCCESS;
}

template<typename T>
Result ReadTypedAttribute(const Tensor& tensor,
                          const std::string& key,
                          const std::string_view expected,
                          std::optional<T>& value) {
    value.reset();

    std::any snapshot;
    JST_CHECK(ReadAttribute(tensor, key, snapshot));
    if (!snapshot.has_value()) {
        return Result::SUCCESS;
    }

    auto* typed = std::any_cast<T>(&snapshot);
    if (!typed) {
        return AttributeTypeError(key, expected, snapshot);
    }

    value = std::move(*typed);
    return Result::SUCCESS;
}

}  // namespace

Result AttributeTypeError(const std::string_view key,
                          const std::string_view expected,
                          const std::any& value) {
    const bool float64 = value.type() == typeid(F64) ||
                         value.type() == typeid(std::vector<F64>);
    const bool pythonHint = float64 && expected.find("F32") != std::string_view::npos;
    JST_ERROR("[MEMORY:TENSOR] Attribute '{}' must have type {} (got {}).{}",
              key, expected, AttributeTypeName(value.type()),
              pythonHint ? " Use np.float32 when assigning Python output attributes." : "");
    return Result::ERROR;
}

Result ValidateAttribute(const std::string_view key, const std::any& value) {
    if (!value.has_value()) {
        JST_ERROR("[MEMORY:TENSOR] Attribute '{}' is present but has no value.", key);
        return Result::ERROR;
    }
    if (key == FrequencyAttribute || key == SampleRateAttribute ||
        key == BandwidthAttribute) {
        return CheckFiniteF32(key, "F32", value);
    }
    if (key == CenterAttribute) {
        if (value.type() == typeid(std::vector<F32>)) {
            return CheckVector<F32>(key, "F32 or vector<F32>", value);
        }
        return CheckFiniteF32(key, "F32 or vector<F32>", value);
    }
    if (key == SampleAxisAttribute || key == BatchAxisAttribute ||
        key == ChannelAxisAttribute) {
        if (value.type() != typeid(Index)) {
            return AttributeTypeError(key, "Index", value);
        }
        return Result::SUCCESS;
    }
    if (key == ChannelOffsetsAttribute) {
        return CheckVector<U64>(key, "vector<U64>", value);
    }
    if (key == ChannelPhaseIncrementsAttribute) {
        return CheckVector<F64>(key, "vector<F64>", value);
    }
    return Result::SUCCESS;
}

Result ReadAttribute(const Tensor& tensor,
                     const std::string& key,
                     std::any& value) {
    value.reset();

    if (!tensor.hasAttribute(key)) {
        return Result::SUCCESS;
    }

    std::any snapshot = tensor.attribute(key);
    JST_CHECK(ValidateAttribute(key, snapshot));

    value = std::move(snapshot);
    return Result::SUCCESS;
}

Result ReadAttribute(const Tensor& tensor,
                     const std::string& key,
                     std::optional<bool>& value) {
    return ReadTypedAttribute(tensor, key, "bool", value);
}

Result ReadAttribute(const Tensor& tensor,
                     const std::string& key,
                     std::optional<F32>& value) {
    return ReadTypedAttribute(tensor, key, "F32", value);
}

Result ReadAttribute(const Tensor& tensor,
                     const std::string& key,
                     std::optional<U64>& value) {
    return ReadTypedAttribute(tensor, key, "U64", value);
}

Result ReadAttribute(const Tensor& tensor,
                     const std::string& key,
                     std::optional<std::vector<U64>>& value) {
    return ReadTypedAttribute(tensor, key, "vector<U64>", value);
}

Result ReadAttribute(const Tensor& tensor,
                     const std::string& key,
                     std::optional<std::vector<F64>>& value) {
    return ReadTypedAttribute(tensor, key, "vector<F64>", value);
}

}  // namespace Jetstream
