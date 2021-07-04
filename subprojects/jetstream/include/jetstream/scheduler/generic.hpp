#ifndef JETSTREAM_SCHED_GENERIC_H
#define JETSTREAM_SCHED_GENERIC_H

#include "jetstream/type.hpp"
#include "jetstream/modules/base.hpp"

namespace Jetstream {

class Scheduler {
public:
    explicit Scheduler(const std::shared_ptr<Module> & m, const Dependencies & d) : mod(m), deps(d) {};
    virtual ~Scheduler() = default;

    virtual Result compute() = 0;
    virtual Result barrier() = 0;

protected:
    const std::shared_ptr<Module> & mod;
    const Dependencies & deps;

    std::atomic<Result> result{SUCCESS};
};

} // namespace Jetstream

#endif
