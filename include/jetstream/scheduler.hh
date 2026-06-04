#ifndef JETSTREAM_SCHEDULER_HH
#define JETSTREAM_SCHEDULER_HH

#include <functional>
#include <memory>
#include <string>
#include <unordered_set>

#include "jetstream/macros.hh"
#include "jetstream/types.hh"

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
    Result start();
    Result stop();
    Result destroy();

    Result add(const std::shared_ptr<Module>& module);
    Result remove(const std::shared_ptr<Module>& module);
    Result reload(const std::shared_ptr<Module>& module);
    Result synchronize(const std::function<Result()>& fn);

    Result present(std::unordered_set<std::string>& failedModules);
    Result compute(std::unordered_set<std::string>& failedModules);

 private:
    std::shared_ptr<Impl> impl;
};

JETSTREAM_API const char* GetSchedulerName(const SchedulerType& scheduler);
JETSTREAM_API const char* GetSchedulerPrettyName(const SchedulerType& scheduler);
JETSTREAM_API SchedulerType StringToScheduler(const std::string& scheduler);

inline std::ostream& operator<<(std::ostream& os, const SchedulerType& scheduler) {
    return os << GetSchedulerPrettyName(scheduler);
}

}  // namespace Jetstream

template <> struct jst::fmt::formatter<Jetstream::SchedulerType> : ostream_formatter {};

#endif  // JETSTREAM_SCHEDULER_HH
