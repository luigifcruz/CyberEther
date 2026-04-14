#ifndef JETSTREAM_RUNTIME_IMPL_HH
#define JETSTREAM_RUNTIME_IMPL_HH

#include <string>

#include "jetstream/runtime.hh"

#include <atomic>

namespace Jetstream {

struct Runtime::Impl {
 public:
    virtual Result create(const Modules& modules) = 0;
    virtual Result destroy() = 0;

    virtual Result compute(const std::vector<std::string>& modules,
                           std::unordered_set<std::string>& skippedModules) = 0;

    virtual const std::shared_ptr<Metrics>& metrics() const = 0;

 protected:
    std::string name;
    DeviceType device;
    RuntimeType backend;
    std::atomic<bool> computeRunning{false};
    std::atomic<bool> presentRunning{false};

    static bool hasSkippedInputs(const std::shared_ptr<Module>& module,
                                 const std::unordered_set<std::string>& skippedModules);

    friend class Runtime;
};

}  // namespace Jetstream

#endif  // JETSTREAM_RUNTIME_IMPL_HH
