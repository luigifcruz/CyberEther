#ifndef JETSTREAM_PARSER_HH
#define JETSTREAM_PARSER_HH

#include <any>
#include <regex>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <filesystem>
#include <unordered_map>

#include "jetstream/types.hh"
#include "jetstream/memory/vector.hh"
#include "jetstream/macros.hh"
#include "jetstream/logger.hh"
#include "jetstream/tools/rapidyaml.hh"

namespace Jetstream {

class Instance;

class Parser {
 public:
    struct Record {
        std::any object;
        U64 hash;
        void* data;
        Locale locale;
        Device device;
        std::string dataType;
        std::vector<U64> shape;
    };

    typedef std::unordered_map<std::string, Record> RecordMap;

    // Block Record

    struct ModuleFingerprint {
        std::string module;
        std::string device;
        std::string dataType;
        std::string inputDataType;
        std::string outputDataType;

        struct Hash {
            U64 operator()(const ModuleFingerprint& m) const {
                U64 h1 = std::hash<std::string>()(m.module);
                U64 h2 = std::hash<std::string>()(m.device);
                U64 h3 = std::hash<std::string>()(m.dataType);
                U64 h4 = std::hash<std::string>()(m.inputDataType);
                U64 h5 = std::hash<std::string>()(m.outputDataType);
                return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3) ^ (h5 << 4);
            }
        };

        struct Equal {
            bool operator()(const ModuleFingerprint& m1, const ModuleFingerprint& m2) const {
                return m1.module == m2.module
                    && m1.device == m2.device
                    && m1.dataType == m2.dataType
                    && m1.inputDataType == m2.inputDataType
                    && m1.outputDataType == m2.outputDataType;
            }
        };

        friend std::ostream& operator<<(std::ostream& os, const ModuleFingerprint& m) {
            os << fmt::format("device: {}, module: {}, ", m.device, m.module);
            if (!m.dataType.empty()) {
                os << fmt::format("dataType: {}", m.dataType);
            } else {
                os << fmt::format("inputDataType: {}, outputDataType: {}", m.inputDataType, m.outputDataType);
            }
            return os;
        }
    };

    struct ModuleRecord {
     public:
        ModuleFingerprint fingerprint;

        Locale locale;
        RecordMap configMap;
        RecordMap inputMap;
        RecordMap outputMap;
        RecordMap interfaceMap;

        void setConfigEndpoint(auto& endpoint) {
            getConfigFunc = [&](RecordMap& map){ return endpoint>>map; };
        }

        void setInputEndpoint(auto& endpoint) {
            getInputFunc = [&](RecordMap& map){ return endpoint>>map; };
        }

        void setOutputEndpoint(auto& endpoint) {
            getOutputFunc = [&](RecordMap& map){ return endpoint>>map; };
        }

        void setInterfaceEndpoint(auto& endpoint) {
            getInterfaceFunc = [&](RecordMap& map){ return endpoint>>map; };
        }

        Result updateMaps() {
            if (getConfigFunc) {
                JST_CHECK(getConfigFunc(configMap));
            }

            if (getInputFunc) {
                JST_CHECK(getInputFunc(inputMap));
            }

            if (getOutputFunc) {
                JST_CHECK(getOutputFunc(outputMap));
            }

            if (getInterfaceFunc) {
                JST_CHECK(getInterfaceFunc(interfaceMap));
            }

            return Result::SUCCESS;
        }

     private:
        std::function<Result(RecordMap& map)> getConfigFunc;
        std::function<Result(RecordMap& map)> getInputFunc;
        std::function<Result(RecordMap& map)> getOutputFunc;
        std::function<Result(RecordMap& map)> getInterfaceFunc;
    };

    // Struct SerDes

    enum class SerDesOp : uint8_t {
        Serialize,
        Deserialize,
    };

    template<typename T>
    static Result Ser(RecordMap& map, const std::string& name, T& variable) {
        if (map.contains(name) != 0) {
            JST_TRACE("Variable name ({}) already inside map. Overwriting.", name);
            map.erase(name);
        }

        auto& metadata = map[name];

        metadata.object = std::any(variable);

        if constexpr (std::is_base_of<VectorType, T>::value) {
            metadata.hash = variable.hash();
            metadata.data = variable.data();
            metadata.locale = variable.locale();
            metadata.device = variable.device();
            metadata.dataType = NumericTypeInfo<typename T::DataType>::name;
            metadata.shape = variable.shape().native();
        }

        return Result::SUCCESS;
    }

    template<typename T>
    static Result Des(RecordMap& map, const std::string& name, T& variable) {
        if (map.contains(name) == 0) {
            JST_WARN("[PARSER] Variable name ({}) not found inside map.", name);
            return Result::SUCCESS;
        }

        auto& anyVar = map[name].object;
        if (!anyVar.has_value()) {
            JST_ERROR("[PARSER] Variable ({}) not initialized.", name);
            return Result::ERROR;
        }

        try {
            DesOpGeneric(anyVar, variable);
            JST_TRACE("Deserializing '{}': Converting std::any to T.", name);
            return Result::SUCCESS;
        } catch (const std::bad_any_cast&) {};

        if constexpr (IsVector<T>::value) {
            try {
                DesOpVector(anyVar, variable);
                JST_TRACE("Deserializing '{}': Converting Vector to T.", name);
                return Result::SUCCESS;
            } catch (const std::bad_any_cast&) {};
        } else {
            try {
                DesOpString(anyVar, variable);
                JST_TRACE("Deserializing '{}': Converting std::string to T.", name);
                return Result::SUCCESS;
            } catch (const std::bad_any_cast&) {};
        }

        JST_ERROR("[PARSER] Failed to cast variable ({}). Check if the input and output are compatible.", name);
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

    // TODO: Add converter from std::any to std::string for export.

    // Configuration File Parser

    Parser();

    Result openFlowgraphFile(const std::string& path);
    Result openFlowgraphBlob(const char* blob);

    Result printFlowgraph() const;

    Result importFlowgraph(Instance& instance);
    Result exportFlowgraph(Instance& instance);

    Result saveFlowgraph(const std::string& path);
    Result closeFlowgraph();

    constexpr bool haveFlowgraph() const {
        return !_fileData.empty();
    }

    constexpr const std::vector<char>& getFlowgraphBlob() const {
        return _fileData;
    }

 private:
    std::vector<char> _fileData;
    ryml::Tree _fileTree;

    static std::vector<char> LoadFile(const std::string& filename);
    static std::vector<std::string> GetMissingKeys(const std::unordered_map<std::string, ryml::ConstNodeRef>& m,
                                                   const std::vector<std::string>& v);
    static std::string GetParameterContents(const std::string& str);
    static std::vector<std::string> GetParameterNodes(const std::string& str);
    static ryml::ConstNodeRef SolvePlaceholder(const ryml::ConstNodeRef& root, const ryml::ConstNodeRef& node);
    static std::unordered_map<std::string, ryml::ConstNodeRef> GatherNodes(const ryml::ConstNodeRef& root,
                                                                           const ryml::ConstNodeRef& node,
                                                                           const std::vector<std::string>& keys,
                                                                           const bool& acceptLess = false);
    static ryml::ConstNodeRef GetNode(const ryml::ConstNodeRef& root, const ryml::ConstNodeRef& node, const std::string& key);
    static bool HasNode(const ryml::ConstNodeRef&, const ryml::ConstNodeRef& node, const std::string& key);
    static std::string ResolveReadable(const ryml::ConstNodeRef& var, const bool& optional = false);
    static std::string ResolveReadableKey(const ryml::ConstNodeRef& var);
    static Record SolveLocalPlaceholder(Instance& instance, const ryml::ConstNodeRef& node);
    static std::vector<std::string> SplitString(const std::string& str, const std::string& delimiter);

    //
    // SerDes Operators
    //

    template<typename T>
    static void DesOpGeneric(std::any& anyVar, T& variable) {
        variable = std::any_cast<T>(anyVar);
    }

    template<Device DeviceId, typename DataType, U64 Dimensions>
    static void DesOpVector(std::any& anyVar, Vector<DeviceId, DataType, Dimensions>& variable) {
        using T = Vector<DeviceId, DataType, Dimensions>;

        if constexpr (DeviceId == Device::CPU) {
            try {
#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
                JST_TRACE("BindVariable: Trying to convert Vector<Metal> into Vector<CPU>.");
                variable = std::move(T(std::any_cast<Vector<Device::Metal, DataType, Dimensions>>(anyVar)));
                return;
#endif
            } catch (const std::bad_any_cast&) {};
        } else if constexpr (DeviceId == Device::Metal) {
            try {
#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
                JST_TRACE("BindVariable: Trying to convert Vector<CPU> into Vector<Metal>.");
                variable = std::move(T(std::any_cast<Vector<Device::CPU, DataType, Dimensions>>(anyVar)));
                return;
#endif
            } catch (const std::bad_any_cast&) {};
        }

        throw std::bad_any_cast();
    }

    // TODO: Maybe add move to all of these?
    template<typename T>
    static void DesOpString(std::any& anyVar, T& variable) {
        if constexpr (std::is_same<T, std::string>::value) {
            JST_TRACE("Converting std::string to std::string.");
            variable = std::any_cast<std::string>(anyVar);
        } else if constexpr (std::is_same<T, U64>::value) {
            JST_TRACE("Converting std::string to U64.");
            variable = std::stoull(std::any_cast<std::string>(anyVar));
        } else if constexpr (std::is_same<T, F32>::value) {
            JST_TRACE("Converting std::string to F32.");
            variable = std::stof(std::any_cast<std::string>(anyVar));
        } else if constexpr (std::is_same<T, F64>::value) {
            JST_TRACE("Converting std::string to F64.");
            variable = std::stod(std::any_cast<std::string>(anyVar));
        } else if constexpr (std::is_same<T, bool>::value) {
            JST_TRACE("Converting std::string to BOOL.");
            std::string lower_s = std::any_cast<std::string>(anyVar);
            std::transform(lower_s.begin(), lower_s.end(), lower_s.begin(), ::tolower);
            variable = lower_s == "true" || lower_s == "1";
        } else if constexpr (std::is_same<T, VectorShape<2>>::value) {
            JST_TRACE("Converting std::string to VectorShape<2>.");
            const auto values = SplitString(std::any_cast<std::string>(anyVar), ", ");
            JST_ASSERT_THROW(values.size() == 2);
            variable = VectorShape<2>{std::stoull(values[0]), std::stoull(values[1])};
        } else if constexpr (std::is_same<T, Range<F32>>::value) {
            JST_TRACE("Converting std::string to Range<F32>.");
            const auto values = SplitString(std::any_cast<std::string>(anyVar), ", ");
            JST_ASSERT_THROW(values.size() == 2);
            variable = Range<F32>{std::stof(values[0]), std::stof(values[1])};
        } else if constexpr (std::is_same<T, Size2D<U64>>::value) {
            JST_TRACE("Converting std::string to Size2D<U64>.");
            const auto values = SplitString(std::any_cast<std::string>(anyVar), ", ");
            JST_ASSERT_THROW(values.size() == 2);
            variable = Size2D<U64>{std::stoull(values[0]), std::stoull(values[1])};
        } else if constexpr (std::is_same<T, Size2D<F32>>::value) {
            JST_TRACE("Converting std::string to Size2D<F32>.");
            const auto values = SplitString(std::any_cast<std::string>(anyVar), ", ");
            JST_ASSERT_THROW(values.size() == 2);
            variable = Size2D<F32>{std::stof(values[0]), std::stof(values[1])};
        } else if constexpr (std::is_same<T, CF32>::value) {
            JST_TRACE("Converting std::string to CF32.");
            variable = StringToComplex<T>(std::any_cast<std::string>(anyVar));
        } else {
            throw std::bad_any_cast();
        }
    }

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
};

}  // namespace Jetstream

template <> struct fmt::formatter<Jetstream::Parser::ModuleFingerprint>   : ostream_formatter {};

#endif
