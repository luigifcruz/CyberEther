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
    struct VectorMetadata {
        U64 hash;
        U64 phash;
        void* ptr;
        Device device;
        std::string type;
        std::vector<U64> shape;
    };

    struct Metadata {
        std::any object;
        VectorMetadata vector;      
    };

    typedef std::unordered_map<std::string, Metadata> RecordMap;

    // Backend Record

    struct BackendIdentifier {
        std::string device;

        struct Hash {
            U64 operator()(const BackendIdentifier& m) const {
                return std::hash<std::string>()(m.device);
            }
        };

        struct Equal {
            bool operator()(const BackendIdentifier& m1, const BackendIdentifier& m2) const {
                return m1.device == m2.device;
            }
        };

        friend std::ostream& operator<<(std::ostream& os, const BackendIdentifier& m) {
            return os << fmt::format("device: {}", m.device);
        }
    };

    struct BackendData {
        RecordMap configMap;
    };

    struct BackendRecord {
        BackendIdentifier id;
        BackendData data;
    };

    // Viewport Record

    struct ViewportIdentifier {
        std::string device;
        std::string platform;

        struct Hash {
            U64 operator()(const ViewportIdentifier& m) const {
                U64 h1 = std::hash<std::string>()(m.device);
                U64 h2 = std::hash<std::string>()(m.platform);
                return h1 ^ (h2 << 1);
            }
        };

        struct Equal {
            bool operator()(const ViewportIdentifier& m1, const ViewportIdentifier& m2) const {
                return m1.device == m2.device
                    && m1.platform == m2.platform;
            }
        };

        friend std::ostream& operator<<(std::ostream& os, const ViewportIdentifier& m) {
            return os << fmt::format("device: {}, platform: {}", m.device, m.platform);
        }
    };

    struct ViewportData {
        RecordMap configMap;
    };

    struct ViewportRecord {
        ViewportIdentifier id;
        ViewportData data;
    };

    // Module Record

    struct ModuleIdentifier {
        std::string module;
        std::string device;
        std::string dataType;
        std::string inputDataType;
        std::string outputDataType;

        struct Hash {
            U64 operator()(const ModuleIdentifier& m) const {
                U64 h1 = std::hash<std::string>()(m.module);
                U64 h2 = std::hash<std::string>()(m.device);
                U64 h3 = std::hash<std::string>()(m.dataType);
                U64 h4 = std::hash<std::string>()(m.inputDataType);
                U64 h5 = std::hash<std::string>()(m.outputDataType);
                return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3) ^ (h5 << 4);
            }
        };

        struct Equal {
            bool operator()(const ModuleIdentifier& m1, const ModuleIdentifier& m2) const {
                return m1.module == m2.module
                    && m1.device == m2.device
                    && m1.dataType == m2.dataType
                    && m1.inputDataType == m2.inputDataType
                    && m1.outputDataType == m2.outputDataType;
            }
        };

        friend std::ostream& operator<<(std::ostream& os, const ModuleIdentifier& m) {
            os << fmt::format("device: {}, module: {} ", m.device, m.module);
            if (!m.dataType.empty()) {
                os << fmt::format("dataType: {}", m.dataType);
            } else {
                os << fmt::format("inputDataType: {}, outputDataType: {}", m.inputDataType, m.outputDataType);
            }
            return os;
        }
    };

    struct ModuleData {
        RecordMap configMap;
        RecordMap inputMap;
        RecordMap outputMap;
        RecordMap interfaceMap;
    };

    struct ModuleRecord {
        std::string name;
        ModuleIdentifier id;
        ModuleData data;
    };

    // Render Record

    struct RenderIdentifier {
        std::string device;

        struct Hash {
            U64 operator()(const RenderIdentifier& m) const {
                return std::hash<std::string>()(m.device);
            }
        };

        struct Equal {
            bool operator()(const RenderIdentifier& m1, const RenderIdentifier& m2) const {
                return m1.device == m2.device;
            }
        };

        friend std::ostream& operator<<(std::ostream& os, const RenderIdentifier& m) {
            return os << fmt::format("device: {}", m.device);
        }
    };

    struct RenderData {
        RecordMap configMap;
    };

    struct RenderRecord {
        RenderIdentifier id;
        RenderData data;
    };

    // Struct SerDes

    enum class SerDesOp : uint8_t {
        Serialize,
        Deserialize,
    };

    template<typename T>
    static Result Ser(RecordMap& map, const std::string& name, const T& variable) {
        if (map.contains(name) != 0) {
            JST_FATAL("Variable name ({}) already inside map.", name);
            return Result::ERROR;
        }

        auto& metadata = map[name];
            
        metadata.object = std::any(variable);

        if constexpr (std::is_base_of<VectorType, T>::value) {
            metadata.vector = {
                variable.hash(),
                variable.phash(),
                variable.data(),
                variable.device(),
                NumericTypeInfo<typename T::DataType>::name,
                variable.shape().native(),
            };
        }

        return Result::SUCCESS;
    }
        
    template<typename T>
    static Result Des(RecordMap& map, const std::string& name, const T& variable) {
        if (map.contains(name) == 0) {
            JST_WARN("Variable name ({}) not found inside map.", name);
            return Result::SUCCESS;
        }

        auto& anyVar = map[name].object;

        if (!anyVar.has_value()) {
            JST_ERROR("Variable ({}) not initialized.", name);
            return Result::ERROR;
        }

        try {
            const_cast<T&>(variable) = std::any_cast<T>(anyVar);
        } catch (const std::bad_any_cast& e) {
            try {
                if constexpr (std::is_same<T, std::string>::value) {
                    JST_TRACE("BindVariable: Converting std::string to std::string.");
                    const_cast<T&>(variable) = std::any_cast<std::string>(anyVar);
                } else if constexpr (std::is_same<T, U64>::value) {
                    JST_TRACE("BindVariable: Converting std::string to U64.");
                    const_cast<T&>(variable) = std::stoull(std::any_cast<std::string>(anyVar));
                } else if constexpr (std::is_same<T, F32>::value) {
                    JST_TRACE("BindVariable: Converting std::string to F32.");
                    const_cast<T&>(variable) = std::stof(std::any_cast<std::string>(anyVar));
                } else if constexpr (std::is_same<T, F64>::value) {
                    JST_TRACE("BindVariable: Converting std::string to F64.");
                    const_cast<T&>(variable) = std::stod(std::any_cast<std::string>(anyVar));
                } else if constexpr (std::is_same<T, bool>::value) {
                    JST_TRACE("BindVariable: Converting std::string to BOOL.");
                    std::string lower_s = std::any_cast<std::string>(anyVar);
                    std::transform(lower_s.begin(), lower_s.end(), lower_s.begin(), ::tolower);
                    const_cast<T&>(variable) = lower_s == "true" || lower_s == "1";
                } else if constexpr (std::is_same<T, VectorShape<2>>::value) {
                    JST_TRACE("BindVariable: Converting std::string to VectorShape<2>.");
                    const auto values = SplitString(std::any_cast<std::string>(anyVar), ", ");
                    JST_ASSERT_THROW(values.size() == 2);
                    const_cast<T&>(variable) = VectorShape<2>{std::stoull(values[0]), std::stoull(values[1])};
                } else if constexpr (std::is_same<T, Range<F32>>::value) {
                    JST_TRACE("BindVariable: Converting std::string to Range<F32>.");
                    const auto values = SplitString(std::any_cast<std::string>(anyVar), ", ");
                    JST_ASSERT_THROW(values.size() == 2);
                    const_cast<T&>(variable) = Range<F32>{std::stof(values[0]), std::stof(values[1])};
                } else if constexpr (std::is_same<T, Size2D<U64>>::value) {
                    JST_TRACE("BindVariable: Converting std::string to Size2D<U64>.");
                    const auto values = SplitString(std::any_cast<std::string>(anyVar), ", ");
                    JST_ASSERT_THROW(values.size() == 2);
                    const_cast<T&>(variable) = Size2D<U64>{std::stoull(values[0]), std::stoull(values[1])};
                } else if constexpr (std::is_same<T, Size2D<F32>>::value) {
                    JST_TRACE("BindVariable: Converting std::string to Size2D<F32>.");
                    const auto values = SplitString(std::any_cast<std::string>(anyVar), ", ");
                    JST_ASSERT_THROW(values.size() == 2);
                    const_cast<T&>(variable) = Size2D<F32>{std::stof(values[0]), std::stof(values[1])};
                } else {
                    const_cast<T&>(variable) = std::any_cast<T>(anyVar);
                }
            } catch (const std::bad_any_cast& e) {
                JST_ERROR("Variable ({}) failed to cast from any. Exhausted cast operators.", name);
                return Result::ERROR;
            }
        }

        return Result::SUCCESS;
    }

    template<typename T>
    static Result SerDes(RecordMap& map, const std::string& name, const T& variable, const SerDesOp& op) {
        if (op == SerDesOp::Deserialize) {
            return Des(map, name, variable);
        } else {
            return Ser(map, name, variable);
        }
    }

    // TODO: Add converter from std::any to std::string for export.

    // Configuration File Parser

    Parser();
    Parser(const std::string& path);

    Result printAll();

    Result importFromFile(Instance& instance);
    Result exportToFile(Instance& instance);
    
    Result createViewport(Instance& instance);
    Result createRender(Instance& instance);
    Result createBackends(Instance& instance);
    Result createModules(Instance& instance);

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
    static std::string ResolveReadable(const ryml::ConstNodeRef& var);
    static std::string ResolveReadableKey(const ryml::ConstNodeRef& var);
    static std::any SolveLocalPlaceholder(Instance& instance, const ryml::ConstNodeRef& node);
    static std::vector<std::string> SplitString(const std::string& str, const std::string& delimiter);
};

}  // namespace Jetstream

template <> struct fmt::formatter<Jetstream::Parser::RenderIdentifier> : ostream_formatter {};
template <> struct fmt::formatter<Jetstream::Parser::BackendIdentifier> : ostream_formatter {};
template <> struct fmt::formatter<Jetstream::Parser::ViewportIdentifier> : ostream_formatter {};
template <> struct fmt::formatter<Jetstream::Parser::ModuleIdentifier> : ostream_formatter {};

#endif