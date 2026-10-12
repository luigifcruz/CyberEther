#include "capture_queue.hh"

#include <algorithm>

#include "jetstream/logger.hh"

namespace Jetstream::Radio {

Result CaptureQueue::resize(const U64 capacity) {
    try {
        std::lock_guard lock(mutex);
        if (storage.size() != capacity) {
            storage.assign(capacity, CF32{});
        }
        head = 0;
        occupancy = 0;
    } catch (const std::exception&) {
        JST_ERROR("[SDR] Failed to allocate the capture queue.");
        return Result::ERROR;
    }
    return Result::SUCCESS;
}

void CaptureQueue::reset(const U64 nextGeneration, const U64 nextDiscard,
                         const F64 nextFrequency, const F64 nextSampleRate) {
    {
        std::lock_guard lock(mutex);
        head = 0;
        occupancy = 0;
        pushed = 0;
        popped = 0;
        gaps.clear();
        generation = nextGeneration;
        discardTransfers = 0;
        discard = nextDiscard;
        frequency = nextFrequency;
        sampleRate = nextSampleRate;
        failure = Device::Status::Samples;
        failureMessage.clear();
        open = true;
    }
    available.notify_all();
}

void CaptureQueue::retune(const U64 nextGeneration, const U64 nextTransfers,
                          const U64 nextDiscard, const F64 nextFrequency) {
    {
        std::lock_guard lock(mutex);
        head = 0;
        occupancy = 0;
        pushed = 0;
        popped = 0;
        gaps.clear();
        generation = nextGeneration;
        discardTransfers = nextTransfers;
        discard = nextDiscard;
        frequency = nextFrequency;
    }
    available.notify_all();
}

void CaptureQueue::relabel(const F64 nextFrequency) {
    std::lock_guard lock(mutex);
    frequency = nextFrequency;
}

void CaptureQueue::close() {
    {
        std::lock_guard lock(mutex);
        open = false;
        head = 0;
        occupancy = 0;
        gaps.clear();
    }
    available.notify_all();
}

void CaptureQueue::push(const U64 sourceGeneration, U64 count, const Converter& convert,
                        const U64 deviceLost) {
    {
        std::lock_guard lock(mutex);
        if (!open || sourceGeneration != generation || failure != Device::Status::Samples) {
            return;
        }
        if (discardTransfers > 0) {
            --discardTransfers;
            return;
        }

        U64 offset = 0;
        if (discard > 0) {
            const U64 skipped = std::min(discard, count);
            discard -= skipped;
            offset = skipped;
            count -= skipped;
        }

        auto addGap = [&](const U64 lost) {
            if (!gaps.empty() && gaps.back().position == pushed) {
                gaps.back().lost += lost;
            } else {
                gaps.push_back({pushed, lost});
            }
        };

        if (deviceLost > 0 && pushed > 0) {
            addGap(deviceLost);
        }

        if (count == 0) {
            return;
        }

        const U64 capacity = storage.size();
        if (capacity == 0 || count > capacity - occupancy) {
            addGap(count);
            return;
        }

        U64 tail = (head + occupancy) % capacity;
        U64 written = 0;
        while (written < count) {
            const U64 chunk = std::min(count - written, capacity - tail);
            convert(storage.data() + tail, offset + written, chunk);
            written += chunk;
            tail = (tail + chunk) % capacity;
        }
        occupancy += count;
        pushed += count;
    }
    available.notify_all();
}

void CaptureQueue::fail(const Device::Status status, const std::string& error) {
    {
        std::lock_guard lock(mutex);
        if (failure == Device::Status::Samples) {
            failure = status;
            failureMessage = error;
        }
    }
    available.notify_all();
}

Device::Capture CaptureQueue::pop(std::span<CF32> samples, const std::chrono::milliseconds timeout) {
    std::unique_lock lock(mutex);
    const auto ready = [&] {
        return occupancy > 0 || !gaps.empty() || failure != Device::Status::Samples || !open;
    };
    if (!available.wait_for(lock, timeout, ready)) {
        return {.status = Device::Status::Timeout, .generation = generation};
    }
    if (!open) {
        return {.status = Device::Status::Failure, .generation = generation,
                .error = "stream is not running"};
    }

    if (!gaps.empty() && gaps.front().position == popped) {
        const auto gap = gaps.front();
        gaps.pop_front();
        return {
            .status = Device::Status::Discontinuity,
            .generation = generation,
            .frequency = frequency,
            .sampleRate = sampleRate,
            .lost = gap.lost,
        };
    }

    if (occupancy == 0) {
        if (failure != Device::Status::Samples) {
            return {.status = failure, .generation = generation, .error = failureMessage};
        }
        return {.status = Device::Status::Timeout, .generation = generation};
    }

    U64 count = std::min<U64>(occupancy, samples.size());
    if (!gaps.empty()) {
        count = std::min(count, gaps.front().position - popped);
    }
    const U64 capacity = storage.size();
    for (U64 i = 0; i < count; ++i) {
        samples[i] = storage[(head + i) % capacity];
    }
    head = (head + count) % capacity;
    occupancy -= count;
    popped += count;

    return {
        .status = Device::Status::Samples,
        .count = count,
        .generation = generation,
        .frequency = frequency,
        .sampleRate = sampleRate,
    };
}

}  // namespace Jetstream::Radio
