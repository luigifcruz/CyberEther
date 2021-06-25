#ifndef JETSTREAM_SCHED_GENERIC_H
#define JETSTREAM_SCHED_GENERIC_H

#include "jetstream/type.hpp"
#include "jetstream/module.hpp"

namespace Jetstream {

class Scheduler {
public:
    explicit Scheduler(const Module & m) : module(m) {};
    virtual ~Scheduler() = default;

    virtual Result compute() = 0;
    virtual Result barrier() = 0;

private:
    const Module& module;

    std::atomic<Result> result{SUCCESS};
};

} // namespace Jetstream

#endif
