#include "stream_buffer.hh"

#include "jetstream/logger.hh"

namespace Jetstream::Radio {

Result StreamBuffer::resize(const U64 capacity) {
    std::lock_guard lock(mutex);
    accumulated = {};
    written = 0;
    breaks.clear();
    expected.reset();
    return samples.resize(capacity);
}

void StreamBuffer::reset(const Metadata& metadata) {
    std::lock_guard lock(mutex);
    const auto stats = samples.statistics();
    accumulated.pushedSamples += stats.pushedElements;
    accumulated.overwrittenSamples += stats.overwrittenElements;
    static_cast<void>(samples.clear());
    current = metadata;
    written = 0;
    breaks.clear();
    expected.reset();
}

bool StreamBuffer::push(const U64 generation, const CF32* data, const U64 count,
                        const std::optional<U64> timestamp, const U64 epoch) {
    std::lock_guard lock(mutex);
    if (generation != current.generation) {
        accumulated.rejectedSamples += count;
        return false;
    }
    if (timestamp) {
        if (expected && (epoch != expectedEpoch || *timestamp != *expected)) {
            mark();
        }
        expected = *timestamp + count;
        expectedEpoch = epoch;
    }
    if (samples.push(data, count) != Result::SUCCESS) {
        return false;
    }
    written += count;
    prune();
    return true;
}

void StreamBuffer::split(const U64 generation) {
    std::lock_guard lock(mutex);
    if (generation != current.generation) {
        return;
    }
    mark();
    expected.reset();
}

Result StreamBuffer::pop(CF32* data, const U64 count, Metadata& metadata) {
    std::lock_guard lock(mutex);
    prune();
    while (!breaks.empty() && breaks.front() - (written - samples.size()) < count) {
        const U64 fragment = breaks.front() - (written - samples.size());
        JST_CHECK(samples.discard(fragment));
        accumulated.discardedSamples += fragment;
        breaks.pop_front();
    }
    if (samples.size() < count) {
        return Result::INCOMPLETE;
    }
    JST_CHECK(samples.pop(data, count));
    metadata = current;
    return Result::SUCCESS;
}

StreamBuffer::Metadata StreamBuffer::metadata() const {
    std::lock_guard lock(mutex);
    return current;
}

U64 StreamBuffer::size() const {
    return samples.size();
}

U64 StreamBuffer::capacity() const {
    return samples.capacity();
}

F64 StreamBuffer::throughput() const {
    return samples.throughput();
}

StreamBuffer::Statistics StreamBuffer::statistics() const {
    std::lock_guard lock(mutex);
    const auto stats = samples.statistics();
    return {
        .pushedSamples = accumulated.pushedSamples + stats.pushedElements,
        .overwrittenSamples = accumulated.overwrittenSamples + stats.overwrittenElements,
        .rejectedSamples = accumulated.rejectedSamples,
        .discardedSamples = accumulated.discardedSamples,
    };
}

void StreamBuffer::mark() {
    if (samples.size() > 0 && (breaks.empty() || breaks.back() != written)) {
        breaks.push_back(written);
    }
}

void StreamBuffer::prune() {
    const U64 head = written - samples.size();
    while (!breaks.empty() && breaks.front() <= head) {
        breaks.pop_front();
    }
}

}  // namespace Jetstream::Radio
