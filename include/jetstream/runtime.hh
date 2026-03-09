#ifndef JETSTREAM_RUNTIME_HH
#define JETSTREAM_RUNTIME_HH

#include <memory>
#include <string>

#include "jetstream/types.hh"

namespace Jetstream {

class Module;

enum class JETSTREAM_API RuntimeType : uint8_t {
    NONE = 0,
    NATIVE,
    MLIR
};

class JETSTREAM_API Runtime {
 public:
    struct Impl;
    struct Context;
    struct Metrics {
        std::string runtime;
        std::string device;
        std::string backend;
        F32 averageComputeTime = 0.0f;
        F32 initializationTime = 0.0f;
        U64 cycles = 0;
    };

    typedef std::unordered_map<std::string, std::shared_ptr<Module>> Modules;

    Runtime(const std::string& name, const DeviceType& device, const RuntimeType& type);

    Result create(const Modules& modules);
    Result destroy();

    Result compute(const std::vector<std::string>& modules = {});

    const std::shared_ptr<Metrics>& metrics() const;

 private:
    std::shared_ptr<Impl> impl;
};

const char* GetRuntimeName(const RuntimeType& runtime);
const char* GetRuntimePrettyName(const RuntimeType& runtime);
RuntimeType StringToRuntime(const std::string& runtime);

inline std::ostream& operator<<(std::ostream& os, const RuntimeType& runtime) {
    return os << GetRuntimePrettyName(runtime);
}

}  // namespace Jetstream

template <> struct jst::fmt::formatter<Jetstream::RuntimeType> : ostream_formatter {};

#endif  // JETSTREAM_RUNTIME_HH
