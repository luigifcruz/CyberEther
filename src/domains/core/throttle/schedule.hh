#ifndef JETSTREAM_DOMAINS_CORE_THROTTLE_SCHEDULE_HH
#define JETSTREAM_DOMAINS_CORE_THROTTLE_SCHEDULE_HH

#include <chrono>

namespace Jetstream::Modules {

struct ThrottleSchedule {
    using Clock = std::chrono::steady_clock;

    Clock::duration interval{};
    Clock::time_point deadline{};

    void reset(const Clock::duration nextInterval, const Clock::time_point now) {
        interval = nextInterval;
        deadline = now;
    }

    void advance(const Clock::time_point now) {
        deadline += interval;
        if (deadline <= now) {
            deadline = now + interval;
        }
    }
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_CORE_THROTTLE_SCHEDULE_HH
