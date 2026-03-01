#ifndef JETSTREAM_SCHEDULER_CONTEXT_HH
#define JETSTREAM_SCHEDULER_CONTEXT_HH

#include "jetstream/scheduler.hh"

namespace Jetstream {

struct Scheduler::Context {
 public:
    virtual Result presentInitialize();
    virtual Result presentSubmit();

    virtual Result hasPendingCompute();
    virtual Result hasPendingPresent();
};

}  // namespace Jetstream

#endif  // JETSTREAM_SCHEDULER_CONTEXT_HH
