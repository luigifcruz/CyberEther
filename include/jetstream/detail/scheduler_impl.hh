#ifndef JETSTREAM_SCHEDULER_IMPL_HH
#define JETSTREAM_SCHEDULER_IMPL_HH

#include <chrono>
#include <functional>
#include <unordered_map>

#include "jetstream/scheduler.hh"

namespace Jetstream {

struct JETSTREAM_API Scheduler::Impl {
 public:
    virtual ~Impl() = default;

    virtual Result create() = 0;
    virtual Result start() = 0;
    virtual Result stop() = 0;
    virtual Result pause() = 0;
    virtual Result resume() = 0;
    virtual bool paused() const = 0;
    virtual Result destroy() = 0;

    virtual Result add(const std::shared_ptr<Module>& module) = 0;
    virtual Result remove(const std::shared_ptr<Module>& module) = 0;
    virtual Result reload(const std::shared_ptr<Module>& module) = 0;
    virtual Result synchronize(const std::function<Result()>& fn) = 0;

    virtual Result present(std::unordered_set<std::string>& failedModules) = 0;
    virtual Result compute(std::unordered_set<std::string>& failedModules) = 0;

    static U64 BeginComputeRound();
    static void ReportComputeProgress();
    static void ReportComputeDeadline(std::chrono::steady_clock::time_point deadline);
    static void WaitComputeRound(U64 epoch, std::chrono::milliseconds timeout);

 protected:
    std::shared_ptr<Instance> instance;

    friend class Scheduler;
};

}  // namespace Jetstream

#endif  // JETSTREAM_SCHEDULER_IMPL_HH
