#ifndef JETSTREAM_SCHED_SYNC_H
#define JETSTREAM_SCHED_SYNC_H

#include "jetstream/scheduler/generic.hpp"

namespace Jetstream {

class Sync : public Scheduler {
public:
    explicit Sync(const std::shared_ptr<Module> &, const Dependencies &);
    ~Sync();

    Result compute();
    Result barrier();

private:
    std::mutex mtx;
    std::atomic<bool> mailbox{false};
};

} // namespace Jetstream

#endif
