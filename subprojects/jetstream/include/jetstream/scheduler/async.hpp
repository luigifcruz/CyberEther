#ifndef JETSTREAM_SCHED_ASYNC_H
#define JETSTREAM_SCHED_ASYNC_H

#include "jetstream/scheduler/generic.hpp"

namespace Jetstream {

class Async : public Scheduler {
public:
    explicit Async(const std::shared_ptr<Module> &, const Dependencies &);
    ~Async();

    Result compute();
    Result barrier();

private:
    std::mutex mtx;
    std::thread worker;
    bool mailbox{false};
    bool discard{false};
    std::condition_variable access;
};

} // namespace Jetstream

#endif
