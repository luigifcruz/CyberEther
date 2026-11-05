#include "stream_buffer.hh"

#include <algorithm>
#include <array>
#include <utility>

#include "jetstream/logger.hh"

namespace Jetstream::Radio {

Result StreamBuffer::resize(const U64 capacity, const U64 channelCount) {
    std::lock_guard lock(mutex);
    if (channelCount == 0) {
        JST_ERROR("[SDR] Stream buffer requires at least one channel.");
        return Result::ERROR;
    }
    std::vector<Tools::CircularBuffer<CF32>> replacement(channelCount);
    for (auto& channel : replacement) {
        JST_CHECK(channel.resize(capacity));
    }
    samples = std::move(replacement);
    accumulated = {};
    written = 0;
    breaks.clear();
    expected.reset();
    return Result::SUCCESS;
}

void StreamBuffer::reset(const Metadata& metadata) {
    std::lock_guard lock(mutex);
    const auto stats = samples.front().statistics();
    accumulated.pushedSamples += stats.pushedElements;
    accumulated.overwrittenSamples += stats.overwrittenElements;
    for (auto& channel : samples) {
        static_cast<void>(channel.clear());
    }
    current = metadata;
    written = 0;
    breaks.clear();
    expected.reset();
}

bool StreamBuffer::push(const U64 generation, const CF32* data, const U64 count,
                        const std::optional<U64> timestamp, const U64 epoch) {
    const std::array<const CF32*, 1> channels{data};
    return push(generation, channels, count, timestamp, epoch);
}

bool StreamBuffer::push(const U64 generation, std::span<const CF32* const> data,
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
        if (samples[i].push(data[i], count) != Result::SUCCESS) {
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

Result StreamBuffer::pop(CF32* data, const U64 count, Metadata& metadata) {
    const std::array<CF32*, 1> channels{data};
    return pop(channels, count, metadata);
}

Result StreamBuffer::pop(std::span<CF32* const> data, const U64 count,
                         Metadata& metadata) {
    std::lock_guard lock(mutex);
    if (data.size() != samples.size() ||
        (count > 0 && std::ranges::find(data, nullptr) != data.end())) {
        JST_ERROR("[SDR] Invalid stream buffer output for {} channels.",
                  samples.size());
        return Result::ERROR;
    }
    prune();
    while (!breaks.empty() &&
           breaks.front() - (written - samples.front().size()) < count) {
        const U64 fragment = breaks.front() - (written - samples.front().size());
        for (auto& channel : samples) {
            JST_CHECK(channel.discard(fragment));
        }
        accumulated.discardedSamples += fragment;
        breaks.pop_front();
    }
    if (samples.front().size() < count) {
        return Result::INCOMPLETE;
    }
    for (U64 i = 0; i < samples.size(); ++i) {
        JST_CHECK(samples[i].pop(data[i], count));
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
    return samples.front().size();
}

U64 StreamBuffer::capacity() const {
    std::lock_guard lock(mutex);
    return samples.front().capacity();
}

F64 StreamBuffer::throughput() const {
    std::lock_guard lock(mutex);
    return samples.front().throughput();
}

StreamBuffer::Statistics StreamBuffer::statistics() const {
    std::lock_guard lock(mutex);
    const auto stats = samples.front().statistics();
    return {
        .pushedSamples = accumulated.pushedSamples + stats.pushedElements,
        .overwrittenSamples = accumulated.overwrittenSamples + stats.overwrittenElements,
        .rejectedSamples = accumulated.rejectedSamples,
        .discardedSamples = accumulated.discardedSamples,
    };
}

void StreamBuffer::mark() {
    if (samples.front().size() > 0 && (breaks.empty() || breaks.back() != written)) {
        breaks.push_back(written);
    }
}

void StreamBuffer::prune() {
    const U64 head = written - samples.front().size();
    while (!breaks.empty() && breaks.front() <= head) {
        breaks.pop_front();
    }
}

}  // namespace Jetstream::Radio
