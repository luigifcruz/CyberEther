#ifndef JETSTREAM_SCHEDULER_HH
#define JETSTREAM_SCHEDULER_HH

#include <memory>
#include <string>

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

    Result present();
    Result compute();

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
