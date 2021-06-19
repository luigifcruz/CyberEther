#ifndef JETSTREAM_SCHED_SYNC_H
#define JETSTREAM_SCHED_SYNC_H

#include "jetstream/scheduler/base.hpp"

namespace Jetstream {

class Sync : public Scheduler {
public:
    explicit Sync(const Graph& d) : Scheduler(d) {};

protected:
    Result start();
    Result end();

    Result compute();
    Result barrier();

private:
    std::mutex m;
    std::atomic<bool> mailbox{false};

    friend class Module;
};

} // namespace Jetstream

#endif
