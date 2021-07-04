#ifndef JETSTREAM_SCHED_H
#define JETSTREAM_SCHED_H

#include "jetstream/scheduler/async.hpp"
#include "jetstream/scheduler/sync.hpp"

namespace Jetstream {

inline std::shared_ptr<Scheduler> SchedulerFactory(const Launch & L, const std::shared_ptr<Module> & m,
        const Dependencies & d) {
    switch (L) {
        case Jetstream::Launch::ASYNC:
            return std::make_shared<Async>(m, d);
        case Jetstream::Launch::SYNC:
            return std::make_shared<Sync>(m, d);
    }
}

} // namespace Jetstream

#endif
