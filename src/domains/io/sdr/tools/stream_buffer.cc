#include "stream_buffer.hh"

#include "jetstream/logger.hh"
#include "jetstream/tools/numeric.hh"

namespace Jetstream::Radio {

Result StreamBuffer::resize(const U64 capacity, const U64 bytes) {
    U64 capacityBytes = 0;
    if (bytes == 0 || !detail::CheckedMultiply(capacity, bytes, capacityBytes)) {
        return Result::ERROR;
    }
    std::lock_guard lock(mutex);
    JST_CHECK(samples.resize(capacityBytes));
    sampleBytes = bytes;
    accumulated = {};
    written = 0;
    breaks.clear();
    expected.reset();
    return Result::SUCCESS;
}

void StreamBuffer::reset(const Metadata& metadata) {
    std::lock_guard lock(mutex);
    const auto stats = samples.statistics();
    accumulated.pushedSamples += stats.pushedElements / sampleBytes;
    accumulated.overwrittenSamples += stats.overwrittenElements / sampleBytes;
    static_cast<void>(samples.clear());
    current = metadata;
    written = 0;
    breaks.clear();
    expected.reset();
}

bool StreamBuffer::push(const U64 generation, const void* data, const U64 count,
                        const std::optional<U64> timestamp, const U64 epoch) {
    std::lock_guard lock(mutex);
    U64 countBytes = 0;
    if (!detail::CheckedMultiply(count, sampleBytes, countBytes)) {
        return false;
    }
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
    if (samples.push(static_cast<const U8*>(data), countBytes) != Result::SUCCESS) {
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

Result StreamBuffer::pop(void* data, const U64 count, Metadata& metadata) {
    std::lock_guard lock(mutex);
    U64 countBytes = 0;
    if (!detail::CheckedMultiply(count, sampleBytes, countBytes)) {
        return Result::ERROR;
    }
    prune();
    while (!breaks.empty() &&
           breaks.front() - (written - samples.size() / sampleBytes) < count) {
        const U64 fragment = breaks.front() - (written - samples.size() / sampleBytes);
        JST_CHECK(samples.discard(fragment * sampleBytes));
        accumulated.discardedSamples += fragment;
        breaks.pop_front();
    }
    if (samples.size() < countBytes) {
        return Result::INCOMPLETE;
    }
    JST_CHECK(samples.pop(static_cast<U8*>(data), countBytes));
    metadata = current;
    return Result::SUCCESS;
}

StreamBuffer::Metadata StreamBuffer::metadata() const {
    std::lock_guard lock(mutex);
    return current;
}

U64 StreamBuffer::size() const {
    std::lock_guard lock(mutex);
    return samples.size() / sampleBytes;
}

U64 StreamBuffer::capacity() const {
    std::lock_guard lock(mutex);
    return samples.capacity() / sampleBytes;
}

F64 StreamBuffer::throughput() const {
    std::lock_guard lock(mutex);
    return samples.throughput() / sampleBytes;
}

StreamBuffer::Statistics StreamBuffer::statistics() const {
    std::lock_guard lock(mutex);
    const auto stats = samples.statistics();
    return {
        .pushedSamples = accumulated.pushedSamples + stats.pushedElements / sampleBytes,
        .overwrittenSamples = accumulated.overwrittenSamples +
                              stats.overwrittenElements / sampleBytes,
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
    const U64 head = written - samples.size() / sampleBytes;
    while (!breaks.empty() && breaks.front() <= head) {
        breaks.pop_front();
    }
}

}  // namespace Jetstream::Radio
