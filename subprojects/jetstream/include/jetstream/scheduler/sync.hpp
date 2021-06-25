#ifndef JETSTREAM_SCHED_SYNC_H
#define JETSTREAM_SCHED_SYNC_H

#include "jetstream/scheduler/generic.hpp"

namespace Jetstream {

class Sync : public Scheduler {
public:
    explicit Sync(const Module & m) : Scheduler(m) {};

    Result compute();
    Result barrier();

private:
    std::mutex m;
    std::atomic<bool> mailbox{false};
};

} // namespace Jetstream

#endif
