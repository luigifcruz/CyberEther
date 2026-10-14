#include "stream_buffer.hh"

#include "jetstream/logger.hh"

namespace Jetstream::Radio {

Result StreamBuffer::resize(const U64 capacity) {
    std::lock_guard lock(mutex);
    accumulated = {};
    return samples.resize(capacity);
}

void StreamBuffer::reset(const Metadata& metadata) {
    std::lock_guard lock(mutex);
    const auto stats = samples.statistics();
    accumulated.pushedSamples += stats.pushedElements;
    accumulated.overwrittenSamples += stats.overwrittenElements;
    static_cast<void>(samples.clear());
    current = metadata;
}

bool StreamBuffer::push(const U64 generation, const CF32* data, const U64 count) {
    std::lock_guard lock(mutex);
    if (generation != current.generation) {
        accumulated.rejectedSamples += count;
        return false;
    }
    return samples.push(data, count) == Result::SUCCESS;
}

Result StreamBuffer::pop(CF32* data, const U64 count, Metadata& metadata) {
    std::lock_guard lock(mutex);
    if (samples.size() < count) {
        return Result::INCOMPLETE;
    }
    JST_CHECK(samples.pop(data, count));
    metadata = current;
    return Result::SUCCESS;
}

Result StreamBuffer::waitForSize(const U64 count, const std::chrono::milliseconds timeout) {
    return samples.waitForSize(count, timeout);
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
    };
}

}  // namespace Jetstream::Radio
