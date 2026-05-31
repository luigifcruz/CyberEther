#ifndef JETSTREAM_RUNTIME_HH
#define JETSTREAM_RUNTIME_HH

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "jetstream/types.hh"

namespace Jetstream {

class Module;

enum class JETSTREAM_API RuntimeType : uint8_t {
    NONE = 0,
    NATIVE = 1 << 0,
    MLIR   = 1 << 1,
};

class JETSTREAM_API Runtime {
 public:
    struct Impl;
    struct Context;

    typedef std::unordered_map<std::string, std::shared_ptr<Module>> Modules;

    Runtime(const std::string& name, const DeviceType& device, const RuntimeType& type);

    Result create(const Modules& modules);
    Result destroy();

    Result compute(const std::vector<std::string>& modules,
                   std::unordered_set<std::string>& skippedModules);

 private:
    std::shared_ptr<Impl> impl;
};

JETSTREAM_API const char* GetRuntimeName(const RuntimeType& runtime);
JETSTREAM_API const char* GetRuntimePrettyName(const RuntimeType& runtime);
JETSTREAM_API RuntimeType StringToRuntime(const std::string& runtime);

inline constexpr RuntimeType operator|(RuntimeType lhs, RuntimeType rhs) {
    return static_cast<RuntimeType>(static_cast<uint8_t>(lhs) | static_cast<uint8_t>(rhs));
}

inline constexpr RuntimeType operator&(RuntimeType lhs, RuntimeType rhs) {
    return static_cast<RuntimeType>(static_cast<uint8_t>(lhs) & static_cast<uint8_t>(rhs));
}

inline std::ostream& operator<<(std::ostream& os, const RuntimeType& runtime) {
    return os << GetRuntimePrettyName(runtime);
}

}  // namespace Jetstream

template <> struct jst::fmt::formatter<Jetstream::RuntimeType> : ostream_formatter {};

#endif  // JETSTREAM_RUNTIME_HH
