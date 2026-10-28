#ifndef JETSTREAM_SCHEDULER_CONTEXT_HH
#define JETSTREAM_SCHEDULER_CONTEXT_HH

#include <chrono>

#include "jetstream/scheduler.hh"

namespace Jetstream {

struct JETSTREAM_API Scheduler::Context {
 public:
    virtual Result presentInitialize();
    virtual Result presentSubmit();

    virtual Result hasPendingCompute();

    static void NotifyPendingCompute();
    static void NotifyPendingComputeAt(std::chrono::steady_clock::time_point deadline);
};

}  // namespace Jetstream

#endif  // JETSTREAM_SCHEDULER_CONTEXT_HH
