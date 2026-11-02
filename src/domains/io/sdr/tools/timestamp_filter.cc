#include "timestamp_filter.hh"

#include <algorithm>
#include <limits>

namespace Jetstream::Radio {

void TimestampFilter::restart(const U64 guardSamples, const U64 maximumStartTimestamp) {
    ++currentEpoch;
    guard = guardSamples;
    maximumStart = maximumStartTimestamp;
    starting = true;
    boundary = 0;
    cursor.reset();
    next.reset();
    gap = false;
}

void TimestampFilter::retune(const U64 validFrom) {
    starting = false;
    boundary = validFrom;
    next.reset();
    gap = false;
}

void TimestampFilter::lose(const U64 validFrom) {
    boundary = std::max(boundary, validFrom);
    gap = next.has_value();
    next.reset();
}

TimestampFilter::Decision TimestampFilter::accept(const U64 timestamp, const U64 count) {
    constexpr U64 Maximum = std::numeric_limits<U64>::max();
    if (timestamp > Maximum - count) {
        return {.status = Device::Status::Failure, .error = "timestamp overflow"};
    }
    if (starting) {
        if (guard == Maximum || timestamp > Maximum - guard) {
            return {.status = Device::Status::Failure, .error = "settling guard or timestamp overflow"};
        }
        if (timestamp > maximumStart) {
            return {
                .status = Device::Status::Failure,
                .error = "stream restart did not reset the hardware timestamp",
            };
        }
        boundary = std::max(boundary, timestamp + guard);
        starting = false;
    }

    const U64 end = timestamp + count;
    cursor = end;
    if (gap) {
        gap = false;
        return {.status = Device::Status::Discontinuity};
    }
    if (end <= boundary) {
        next.reset();
        return {.status = Device::Status::Timeout, .skip = count};
    }
    if (next && timestamp != *next) {
        const U64 lost = timestamp > *next ? timestamp - *next : 0;
        next = end;
        return {.status = Device::Status::Discontinuity, .lost = lost};
    }
    next = end;
    return {.status = Device::Status::Samples,
            .skip = timestamp < boundary ? boundary - timestamp : 0};
}

U64 TimestampFilter::pending() const {
    if (starting || !cursor || *cursor >= boundary) {
        return 0;
    }
    return boundary - *cursor;
}

U64 TimestampFilter::epoch() const {
    return currentEpoch;
}

}  // namespace Jetstream::Radio
