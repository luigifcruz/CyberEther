#ifndef JETSTREAM_SCHED_ASYNC_H
#define JETSTREAM_SCHED_ASYNC_H

#include "jetstream/scheduler/base.hpp"

namespace Jetstream {

class Async : public Scheduler {
public:
    explicit Async(const Graph& d) : Scheduler(d) {};

protected:
    Result start();
    Result end();

    Result compute();
    Result barrier();

private:
    std::mutex m;
    std::thread worker;
    bool mailbox{false};
    bool discard{false};
    std::condition_variable access;

    friend class Module;
};

} // namespace Jetstream

#endif
