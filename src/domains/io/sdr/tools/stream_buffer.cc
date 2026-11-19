#include "stream_buffer.hh"

#include <algorithm>
#include <array>
#include <utility>

#include "jetstream/logger.hh"
#include "jetstream/tools/numeric.hh"

namespace Jetstream::Radio {

Result StreamBuffer::resize(const U64 capacity, const U64 channels, const U64 bytes) {
    std::lock_guard lock(mutex);
    if (channels == 0) {
        JST_ERROR("[SDR] Stream buffer requires at least one channel.");
        return Result::ERROR;
    }
    U64 capacityBytes = 0;
    if (bytes == 0 || !detail::CheckedMultiply(capacity, bytes, capacityBytes)) {
        JST_ERROR("[SDR] Stream buffer size is too large.");
        return Result::ERROR;
    }
    std::vector<Tools::CircularBuffer<U8>> replacement(channels);
    for (auto& channel : replacement) {
        JST_CHECK(channel.resize(capacityBytes));
    }
    samples = std::move(replacement);
    sampleBytes = bytes;
    accumulated = {};
    written = 0;
    breaks.clear();
    expected.reset();
    return Result::SUCCESS;
}

void StreamBuffer::reset(const Metadata& metadata) {
    std::lock_guard lock(mutex);
    const auto stats = samples.front().statistics();
    accumulated.pushedSamples += stats.pushedElements / sampleBytes;
    accumulated.overwrittenSamples += stats.overwrittenElements / sampleBytes;
    for (auto& channel : samples) {
        static_cast<void>(channel.clear());
    }
    current = metadata;
    written = 0;
    breaks.clear();
    expected.reset();
}

bool StreamBuffer::push(const U64 generation, const void* data, const U64 count,
                        const std::optional<U64> timestamp, const U64 epoch) {
    const std::array<const void*, 1> channels{data};
    return push(generation, channels, count, timestamp, epoch);
}

bool StreamBuffer::push(const U64 generation, std::span<const void* const> data,
                        const U64 count, const std::optional<U64> timestamp,
                        const U64 epoch) {
    std::lock_guard lock(mutex);
    if (data.size() != samples.size() ||
        (count > 0 && std::ranges::find(data, nullptr) != data.end())) {
        JST_ERROR("[SDR] Invalid stream buffer input for {} channels.", samples.size());
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
    for (U64 i = 0; i < samples.size(); ++i) {
        if (samples[i].push(static_cast<const U8*>(data[i]), count * sampleBytes) !=
            Result::SUCCESS) {
            return false;
        }
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
    const std::array<void*, 1> channels{data};
    return pop(channels, count, metadata);
}

Result StreamBuffer::pop(std::span<void* const> data, const U64 count,
                         Metadata& metadata) {
    std::lock_guard lock(mutex);
    if (data.size() != samples.size() ||
        (count > 0 && std::ranges::find(data, nullptr) != data.end())) {
        JST_ERROR("[SDR] Invalid stream buffer output for {} channels.",
                  samples.size());
        return Result::ERROR;
    }
    prune();
    while (!breaks.empty() && breaks.front() - (written - stored()) < count) {
        const U64 fragment = breaks.front() - (written - stored());
        for (auto& channel : samples) {
            JST_CHECK(channel.discard(fragment * sampleBytes));
        }
        accumulated.discardedSamples += fragment;
        breaks.pop_front();
    }
    if (stored() < count) {
        return Result::INCOMPLETE;
    }
    for (U64 i = 0; i < samples.size(); ++i) {
        JST_CHECK(samples[i].pop(static_cast<U8*>(data[i]), count * sampleBytes));
    }
    metadata = current;
    return Result::SUCCESS;
}

StreamBuffer::Metadata StreamBuffer::metadata() const {
    std::lock_guard lock(mutex);
    return current;
}

U64 StreamBuffer::size() const {
    std::lock_guard lock(mutex);
    return stored();
}

U64 StreamBuffer::capacity() const {
    std::lock_guard lock(mutex);
    return samples.front().capacity() / sampleBytes;
}

F64 StreamBuffer::throughput() const {
    std::lock_guard lock(mutex);
    return samples.front().throughput() / static_cast<F64>(sampleBytes);
}

StreamBuffer::Statistics StreamBuffer::statistics() const {
    std::lock_guard lock(mutex);
    const auto stats = samples.front().statistics();
    return {
        .pushedSamples = accumulated.pushedSamples + stats.pushedElements / sampleBytes,
        .overwrittenSamples = accumulated.overwrittenSamples +
                              stats.overwrittenElements / sampleBytes,
        .rejectedSamples = accumulated.rejectedSamples,
        .discardedSamples = accumulated.discardedSamples,
    };
}

U64 StreamBuffer::stored() const {
    return samples.front().size() / sampleBytes;
}

void StreamBuffer::mark() {
    if (stored() > 0 && (breaks.empty() || breaks.back() != written)) {
        breaks.push_back(written);
    }
}

void StreamBuffer::prune() {
    const U64 head = written - stored();
    while (!breaks.empty() && breaks.front() <= head) {
        breaks.pop_front();
    }
}

}  // namespace Jetstream::Radio
