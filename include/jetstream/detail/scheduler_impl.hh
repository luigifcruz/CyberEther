#ifndef JETSTREAM_SCHEDULER_IMPL_HH
#define JETSTREAM_SCHEDULER_IMPL_HH

#include <unordered_map>

#include "jetstream/scheduler.hh"

namespace Jetstream {

struct Scheduler::Impl {
 public:
    virtual ~Impl() = default;

    virtual Result create() = 0;
    virtual Result destroy() = 0;

    virtual Result add(const std::shared_ptr<Module>& module) = 0;
    virtual Result remove(const std::shared_ptr<Module>& module) = 0;
    virtual Result reload(const std::shared_ptr<Module>& module) = 0;

    virtual Result present() = 0;
    virtual Result compute() = 0;

    virtual const std::unordered_map<std::string, std::shared_ptr<Runtime::Metrics>>& metrics() const = 0;

 protected:
    std::shared_ptr<Instance> instance;

    friend class Scheduler;
};

}  // namespace Jetstream

#endif  // JETSTREAM_SCHEDULER_IMPL_HH
