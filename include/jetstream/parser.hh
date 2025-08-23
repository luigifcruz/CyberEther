#ifndef JETSTREAM_PARSER_HH
#define JETSTREAM_PARSER_HH

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <sstream>
#include <iostream>
#include <unordered_map>
#include <unordered_set>

#include "jetstream/types.hh"
#include "jetstream/macros.hh"
#include "jetstream/logger.hh"
#include "jetstream/memory/base.hh"

namespace Jetstream {

class Instance;

class Parser {
 public:
    struct Record {
        std::any object;
        U64 hash = 0;
        void* data = nullptr;
        Locale locale = {};
        Device device = Device::None;
        std::string dataType = "";
        std::vector<U64> shape = {};
        std::map<std::string, std::string> attributes = {};
        bool host_accessible = false;
        bool device_native = false;
        bool host_native = false;
        bool contiguous = false;
    };

    typedef std::unordered_map<std::string, Record> RecordMap;

    enum class SerDesOp : uint8_t {
        Serialize,
        Deserialize,
    };

    struct Adapter {
     public:
        virtual ~Adapter() = default;

        virtual Result deserialize(const std::any& var) = 0;
        virtual Result serialize(std::any& var) const = 0;
    };

    template<typename T>
    static Result Ser(RecordMap& map, const std::string& name, T& variable) {
        if (map.contains(name) != 0) {
            JST_TRACE("Variable name '{}' already inside map. Overwriting.", name);
            map.erase(name);
        }

        auto& metadata = map[name];

        if constexpr (std::is_base_of<Adapter, T>::value) {
            variable.serialize(metadata.object);
        } else {
            metadata.object = std::any(variable);
        }

        if constexpr (IsTensor<T>::value) {
            metadata.hash = variable.hash();
            metadata.data = variable.data();
            metadata.device = variable.device();
            metadata.dataType = NumericTypeInfo<typename T::DataType>::name;
            metadata.shape = variable.shape();
            metadata.locale = variable.locale();
            metadata.host_accessible = variable.host_accessible();
            metadata.device_native = variable.device_native();
            metadata.host_native = variable.host_native();
            metadata.contiguous = variable.contiguous();

            for (const auto& [key, attribute] : variable.attributes()) {
                JST_CHECK(AnyToString(attribute.get(), metadata.attributes[key], true));
            }
        }

        return Result::SUCCESS;
    }

    template<typename T>
    static Result Des(RecordMap& map, const std::string& name, T& variable) {
        if (map.contains(name) == 0) {
            JST_TRACE("[PARSER] Variable name '{}' not found inside map.", name);
            return Result::SUCCESS;
        }

        auto& anyVar = map.at(name).object;
        if (!anyVar.has_value()) {
            JST_ERROR("[PARSER] Variable '{}' not initialized.", name);
            return Result::ERROR;
        }

        if (anyVar.type() == typeid(T)) {
            JST_TRACE("Deserializing '{}': Trying to convert 'std::any' into 'T'.", name);

            variable = std::move(std::any_cast<T>(anyVar));
            return Result::SUCCESS;
        }

        if constexpr (std::is_base_of<Adapter, T>::value) {
            JST_TRACE("Deserializing '{}': Trying to convert 'std::any' into 'T' with custom deserialize method.", name);
            JST_CHECK(variable.deserialize(anyVar));
            return Result::SUCCESS;
        }

        if constexpr (IsTensor<T>::value) {
            JST_TRACE("Deserializing '{}': Trying to convert 'std::any' into 'Tensor'.", name);

            if (map.at(name).locale.empty()) {
                JST_TRACE("Deserializing '{}': Tensor has no locale. Skipping.", name);
                return Result::SUCCESS;
            }

            if (variable.device() == Device::CPU) {
#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
                if (anyVar.type() == typeid(Tensor<Device::Metal, typename T::DataType>)) {
                    return SafeTensorCast<Device::Metal>(name, anyVar, variable);
                }
#endif
#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
                if (anyVar.type() == typeid(Tensor<Device::CUDA, typename T::DataType>)) {
                    return SafeTensorCast<Device::CUDA>(name, anyVar, variable);
                }
#endif
            } else if (variable.device() == Device::Metal) {
#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
                if (anyVar.type() == typeid(Tensor<Device::CPU, typename T::DataType>)) {
                    return SafeTensorCast<Device::CPU>(name, anyVar, variable);
                }
#endif
            } else if (variable.device() == Device::CUDA) {
#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
                if (anyVar.type() == typeid(Tensor<Device::CPU, typename T::DataType>)) {
                    return SafeTensorCast<Device::CPU>(name, anyVar, variable);
                }
#endif
            }

            JST_ERROR("[PARSER] Failed to cast Tensor. Check if the type and device are compatible.");
            JST_TRACE("[PARSER] Variable type: {}", anyVar.type().name());
            return Result::ERROR;
        }

        if constexpr (std::is_same<T, std::string>::value) {
            JST_TRACE("Deserializing '{}': Trying to convert 'std::any' into 'std::string'.", name);

            if (anyVar.type() == typeid(std::string)) {
                variable = std::move(std::any_cast<std::string>(anyVar));
                return Result::SUCCESS;
            }
        }

        if constexpr (std::is_same<T, I32>::value) {
            JST_TRACE("Deserializing '{}': Trying to convert 'std::any' into 'I32'.", name);

            if (anyVar.type() == typeid(std::string)) {
                variable = std::move(std::stoi(std::any_cast<std::string>(anyVar)));
                return Result::SUCCESS;
            }
        }

        if constexpr (std::is_same<T, U64>::value) {
            JST_TRACE("Deserializing '{}': Trying to convert 'std::any' into 'U64'.", name);

            if (anyVar.type() == typeid(std::string)) {
                variable = std::move(std::stoull(std::any_cast<std::string>(anyVar)));
                return Result::SUCCESS;
            }
        }

        if constexpr (std::is_same<T, F32>::value) {
            JST_TRACE("Deserializing '{}': Trying to convert 'std::any' into 'F32'.", name);

            if (anyVar.type() == typeid(std::string)) {
                variable = std::move(std::stof(std::any_cast<std::string>(anyVar)));
                return Result::SUCCESS;
            }
        }

        if constexpr (std::is_same<T, F64>::value) {
            JST_TRACE("Deserializing '{}': Trying to convert 'std::any' into 'F64'.", name);

            if (anyVar.type() == typeid(std::string)) {
                variable = std::move(std::stod(std::any_cast<std::string>(anyVar)));
                return Result::SUCCESS;
            }
        }

        if constexpr (std::is_same<T, CF32>::value) {
            JST_TRACE("Deserializing '{}': Trying to convert 'std::any' into 'CF32'.", name);

            if (anyVar.type() == typeid(std::string)) {
                variable = std::move(StringToComplex<T>(std::any_cast<std::string>(anyVar)));
                return Result::SUCCESS;
            }
        }

        if constexpr (std::is_same<T, bool>::value) {
            JST_TRACE("Deserializing '{}': Trying to convert 'std::any' into 'BOOL'.", name);

            if (anyVar.type() == typeid(std::string)) {
                std::string lower_s = std::any_cast<std::string>(anyVar);
                std::transform(lower_s.begin(), lower_s.end(), lower_s.begin(), ::tolower);
                variable = std::move(lower_s == "true" || lower_s == "1");
                return Result::SUCCESS;
            }
        }

        if constexpr (std::is_same<T, std::vector<U64>>::value) {
            JST_TRACE("Deserializing '{}': Trying to convert 'std::any' into 'std::vector<U64>'.", name);

            if (anyVar.type() == typeid(std::string)) {
                const auto& values = SplitString(std::any_cast<std::string>(anyVar), ", ");
                variable = std::move(std::vector<U64>(values.size()));
                std::transform(values.begin(), values.end(), variable.begin(), [](const std::string& s){ return std::stoull(s); });
                return Result::SUCCESS;
            }
        }

        if constexpr (std::is_same<T, std::vector<F32>>::value) {
            JST_TRACE("Deserializing '{}': Trying to convert 'std::any' into 'std::vector<F32>'.", name);

            if (anyVar.type() == typeid(std::string)) {
                const auto& values = SplitString(std::any_cast<std::string>(anyVar), ", ");
                variable = std::move(std::vector<F32>(values.size()));
                std::transform(values.begin(), values.end(), variable.begin(), [](const std::string& s){ return std::stof(s); });
                return Result::SUCCESS;
            }
        }

        if constexpr (std::is_same<T, Range<F32>>::value) {
            JST_TRACE("Deserializing '{}': Trying to convert 'std::any' into 'Range<F32>'.", name);

            if (anyVar.type() == typeid(std::string)) {
                const auto& values = SplitString(std::any_cast<std::string>(anyVar), ", ");
                JST_ASSERT(values.size() == 2, "Unexpected number of values.");
                variable = std::move(Range<F32>{std::stof(values[0]), std::stof(values[1])});
                return Result::SUCCESS;
            }
        }

        if constexpr (std::is_same<T, Extent2D<U64>>::value) {
            JST_TRACE("Deserializing '{}': Trying to convert 'std::any' into 'Extent2D<U64>'.", name);

            if (anyVar.type() == typeid(std::string)) {
                const auto& values = SplitString(std::any_cast<std::string>(anyVar), ", ");
                JST_ASSERT(values.size() == 2, "Unexpected number of values.");
                variable = std::move(Extent2D<U64>{std::stoull(values[0]), std::stoull(values[1])});
                return Result::SUCCESS;
            }
        }

        if constexpr (std::is_same<T, Extent2D<F32>>::value) {
            JST_TRACE("Deserializing '{}': Trying to convert 'std::any' into 'Extent2D<F32>'.", name);

            if (anyVar.type() == typeid(std::string)) {
                const auto& values = SplitString(std::any_cast<std::string>(anyVar), ", ");
                JST_ASSERT(values.size() == 2, "Unexpected number of values.");
                variable = std::move(Extent2D<F32>{std::stof(values[0]), std::stof(values[1])});
                return Result::SUCCESS;
            }
        }

        // [TYPE SERIALIZATION HOOK]

        JST_ERROR("[PARSER] Failed to cast variable '{}'. Check if the input and output are compatible.", name);
        return Result::ERROR;
    }

    template<typename T>
    static Result SerDes(RecordMap& map, const std::string& name, T& variable, const SerDesOp& op) {
        if (op == SerDesOp::Deserialize) {
            return Des(map, name, variable);
        } else {
            return Ser(map, name, variable);
        }
    }

    static Result SerializeToString(const Record& record, std::string& out) {
        if (record.device != Device::None) {
            if (!record.locale.empty()) {
                out = jst::fmt::format("${{graph.{}.output.{}}}", record.locale.blockId, record.locale.pinId);
                return Result::SUCCESS;
            }

            return Result::SUCCESS;
        }

        JST_CHECK(AnyToString(record.object, out));

        return Result::SUCCESS;
    }

    static Result AnyToString(const std::any& var, std::string& out, const bool& optional = false) {
        if (var.type() == typeid(std::string)) {
            const auto& string_value = std::any_cast<std::string>(var);

            if (std::count(string_value.begin(), string_value.end(), '\n')) {
                out = jst::fmt::format("{}", string_value);
            } else {
                out = jst::fmt::format("'{}'", string_value);
            }

            return Result::SUCCESS;
        }

        if (var.type() == typeid(I32)) {
            out = jst::fmt::format("{}", std::any_cast<I32>(var));
            return Result::SUCCESS;
        }

        if (var.type() == typeid(U64)) {
            out = jst::fmt::format("{}", std::any_cast<U64>(var));
            return Result::SUCCESS;
        }

        if (var.type() == typeid(F32)) {
            out = jst::fmt::format("{}", std::any_cast<F32>(var));
            return Result::SUCCESS;
        }

        if (var.type() == typeid(F64)) {
            out = jst::fmt::format("{}", std::any_cast<F64>(var));
            return Result::SUCCESS;
        }

        if (var.type() == typeid(CF32)) {
            const auto& complex = std::any_cast<CF32>(var);
            out = jst::fmt::format("{}{}{}", complex.real(), complex.imag() < 0 ? "-" : "+", complex.imag());
            return Result::SUCCESS;
        }

        if (var.type() == typeid(bool)) {
            out = jst::fmt::format("{}", std::any_cast<bool>(var));
            return Result::SUCCESS;
        }

        if (var.type() == typeid(std::vector<U64>)) {
            const auto& values = std::any_cast<std::vector<U64>>(var);
            std::stringstream ss;
            std::copy(values.begin(), values.end(), std::ostream_iterator<U64>(ss, ", "));
            out = jst::fmt::format("[{}]", ss.str());
            return Result::SUCCESS;
        }

        if (var.type() == typeid(std::vector<F32>)) {
            const auto& values = std::any_cast<std::vector<F32>>(var);
            std::stringstream ss;
            std::copy(values.begin(), values.end(), std::ostream_iterator<F32>(ss, ", "));
            out = jst::fmt::format("[{}]", ss.str());
            return Result::SUCCESS;
        }

        if (var.type() == typeid(Range<F32>)) {
            const auto& range = std::any_cast<Range<F32>>(var);
            out = jst::fmt::format("[{}, {}]", range.min, range.max);
            return Result::SUCCESS;
        }

        if (var.type() == typeid(Extent2D<U64>)) {
            const auto& size = std::any_cast<Extent2D<U64>>(var);
            out = jst::fmt::format("[{}, {}]", size.x, size.y);
            return Result::SUCCESS;
        }

        if (var.type() == typeid(Extent2D<F32>)) {
            const auto& size = std::any_cast<Extent2D<F32>>(var);
            out = jst::fmt::format("[{}, {}]", size.x, size.y);
            return Result::SUCCESS;
        }

        // [TYPE SERIALIZATION HOOK]

        if (!optional) {
            JST_ERROR("[PARSER] Failed to serialize variable. Check if the input and output are compatible.");
            JST_TRACE("[PARSER] Variable type: {}", var.type().name());
            return Result::ERROR;
        }

        return Result::SUCCESS;
    }

    static std::vector<std::string> SplitString(const std::string& str, const std::string& delimiter);

 private:
    template<typename T>
    static T StringToComplex(const std::string& s) {
        using ST = typename NumericTypeInfo<T>::subtype;

        ST real = 0.0;
        ST imag = 0.0;
        char op = '+';

        std::stringstream ss(s);
        ss >> real;      // Extract real part
        ss >> op;        // Extract '+' or '-'
        ss >> imag;      // Extract imaginary part

        if (op == '-') {
            imag = -imag;
        }

        return T(real, imag);
    }

    template<Device SrcD, Device DstD, typename T>
    static Result SafeTensorCast(const std::string& name, std::any& anyVar, Tensor<DstD, T>& variable) {
        (void)name;

        JST_TRACE("Deserializing '{}': Trying to convert 'Tensor<Device::{}>' into 'Tensor<Device::{}>'.", name, SrcD, DstD);
        const auto& tensor = std::any_cast<Tensor<SrcD, T>>(anyVar);

        if (!tensor.compatible_devices().contains(DstD)) {
            JST_ERROR("[PARSER] Failed to cast Tensor device from {} to {}.", SrcD, DstD);
            JST_TRACE("[PARSER] Supported casts: {}", tensor.compatible_devices());
            return Result::ERROR;
        }

        variable = std::move(Tensor<DstD, T>(tensor));
        return Result::SUCCESS;
    }
};

}  // namespace Jetstream

#endif
