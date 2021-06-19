#ifndef JETSTREAM_SCHED_H
#define JETSTREAM_SCHED_H

#include "jetstream/type.hpp"

namespace Jetstream {

class Scheduler {
public:
    explicit Scheduler(const Graph& d) : deps(d) {};
    virtual ~Scheduler() = default;

protected:
    virtual Result start() = 0;
    virtual Result end() = 0;

    virtual Result compute() = 0;
    virtual Result barrier() = 0;

    virtual Result underlyingCompute() = 0;

    const Graph& deps;

    std::atomic<Result> result{SUCCESS};
};

} // namespace Jetstream

#endif
