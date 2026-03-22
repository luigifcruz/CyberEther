#ifndef JETSTREAM_SCHEDULER_HH
#define JETSTREAM_SCHEDULER_HH

#include <memory>
#include <string>
#include <unordered_map>

#include "jetstream/macros.hh"
#include "jetstream/types.hh"
#include "jetstream/runtime.hh"

namespace Jetstream {

class Instance;
class Module;

enum class JETSTREAM_API SchedulerType : uint8_t {
    NONE = 0,
    SYNCHRONOUS,
};

class JETSTREAM_API Scheduler {
 public:
    struct Impl;
    struct Context;

    Scheduler(const SchedulerType& type);

    Result create(const std::shared_ptr<Instance>& instance);
    Result destroy();

    Result add(const std::shared_ptr<Module>& module);
    Result remove(const std::shared_ptr<Module>& module);
    Result reload(const std::shared_ptr<Module>& module);

    Result present();
    Result compute();

    const std::unordered_map<std::string, std::shared_ptr<Runtime::Metrics>>& metrics() const;

 private:
    std::shared_ptr<Impl> impl;
};

const char* GetSchedulerName(const SchedulerType& scheduler);
const char* GetSchedulerPrettyName(const SchedulerType& scheduler);
SchedulerType StringToScheduler(const std::string& scheduler);

inline std::ostream& operator<<(std::ostream& os, const SchedulerType& scheduler) {
    return os << GetSchedulerPrettyName(scheduler);
}

}  // namespace Jetstream

template <> struct jst::fmt::formatter<Jetstream::SchedulerType> : ostream_formatter {};

#endif  // JETSTREAM_SCHEDULER_HH
