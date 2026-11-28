#include "instance_remote_audio.hh"

#include <algorithm>
#include <array>
#include <cmath>

namespace Jetstream {

namespace {

constexpr U64 kSourceCapacityFrames = Instance::Remote::Audio::SampleRate / 2;
constexpr U64 kPushChunkFrames = 512;

}  // namespace

Instance::Remote::Audio::Source::Source() : buffer(kSourceCapacityFrames * Channels) {}

Instance::Remote::Audio::Mixer& Instance::Remote::Audio::Shared() {
    static auto* mixer = new Mixer();
    return *mixer;
}

bool Instance::Remote::Audio::Active() {
    return Shared().active.load(std::memory_order_relaxed);
}

std::shared_ptr<Instance::Remote::Audio::Source> Instance::Remote::Audio::Attach() {
    auto source = std::make_shared<Source>();
    auto& mixer = Shared();
    std::lock_guard<std::mutex> lock(mixer.mutex);
    std::erase_if(mixer.sources, [](const auto& entry) {
        return entry.expired();
    });
    mixer.sources.push_back(source);
    return source;
}

void Instance::Remote::Audio::Push(Source& source, const F32* frames, const U64 frameCount, const U32 channels,
                                   const F32 gain) {
    if (!Active() || !frames || frameCount == 0 || (channels != 1 && channels != 2)) {
        return;
    }

    std::array<F32, kPushChunkFrames * Channels> scratch;
    for (U64 offset = 0; offset < frameCount;) {
        const U64 count = std::min(frameCount - offset, kPushChunkFrames);
        for (U64 i = 0; i < count; ++i) {
            const F32* frame = frames + (offset + i) * channels;
            const F32 left = frame[0] * gain;
            const F32 right = frame[channels - 1] * gain;
            scratch[i * Channels] = std::isfinite(left) ? left : 0.0f;
            scratch[i * Channels + 1] = std::isfinite(right) ? right : 0.0f;
        }
        (void)source.buffer.push(scratch.data(), count * Channels);
        offset += count;
    }
}

}  // namespace Jetstream
