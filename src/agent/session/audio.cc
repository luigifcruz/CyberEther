#include "jetstream/agent/session.hh"

#include "jetstream/config.hh"
#include "jetstream/logger.hh"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <mutex>

#if defined(JETSTREAM_AGENT_LIVE_AVAILABLE)
#include "tools/miniaudio/miniaudio.h"
#include "tools/base64.hh"
#endif

namespace Jetstream::Agent {

#if defined(JETSTREAM_AGENT_LIVE_AVAILABLE)

namespace {

constexpr U64 kMaxPlaybackSeconds = 30;

F32 RmsLevel(const I16* samples, const U64 count) {
    if (count == 0) {
        return 0.0f;
    }

    F64 sum = 0.0;
    for (U64 i = 0; i < count; ++i) {
        const F64 value = static_cast<F64>(samples[i]) / 32768.0;
        sum += value * value;
    }

    return static_cast<F32>(std::sqrt(sum / static_cast<F64>(count)));
}

void SmoothLevel(std::atomic<F32>& level, const F32 next) {
    const F32 previous = level.load(std::memory_order_relaxed);
    const F32 rate = next > previous ? 0.6f : 0.15f;
    level.store(previous + (next - previous) * rate, std::memory_order_relaxed);
}

}  // namespace

struct Session::Audio::Impl {
    explicit Impl(U32 rate) : sampleRate(rate) {}

    U32 sampleRate;
    ma_device device{};
    bool initialized = false;
    std::mutex captureMutex;
    std::vector<I16> captureQueue;
    std::mutex playbackMutex;
    std::vector<I16> playbackQueue;
    U64 playbackOffset = 0;
    std::atomic<F32> input{0.0f};
    std::atomic<F32> output{0.0f};

    static void callback(ma_device* device, void* output, const void* input, ma_uint32 frameCount) {
        auto* self = static_cast<Impl*>(device->pUserData);
        if (input) {
            const auto* samples = static_cast<const I16*>(input);
            SmoothLevel(self->input, RmsLevel(samples, frameCount));

            std::lock_guard<std::mutex> lock(self->captureMutex);
            self->captureQueue.insert(self->captureQueue.end(), samples, samples + frameCount);
        }

        if (output) {
            auto* samples = static_cast<I16*>(output);
            U64 copied = 0;
            {
                std::lock_guard<std::mutex> lock(self->playbackMutex);
                const U64 available = self->playbackQueue.size() - self->playbackOffset;
                copied = std::min<U64>(available, frameCount);
                if (copied > 0) {
                    const auto* source = self->playbackQueue.data() + self->playbackOffset;
                    std::memcpy(samples, source, copied * sizeof(I16));
                    self->playbackOffset += copied;
                }

                if (self->playbackOffset >= self->playbackQueue.size()) {
                    self->playbackQueue.clear();
                    self->playbackOffset = 0;
                }
            }

            if (copied < frameCount) {
                std::memset(samples + copied, 0, (frameCount - copied) * sizeof(I16));
            }

            SmoothLevel(self->output, RmsLevel(samples, copied));
        }
    }
};

Session::Audio::Audio(U32 sampleRate) : impl(std::make_unique<Impl>(sampleRate)) {}

Session::Audio::~Audio() {
    stop();
}

Result Session::Audio::start() {
    ma_device_config config = ma_device_config_init(ma_device_type_duplex);
    config.sampleRate = impl->sampleRate;
    config.capture.format = ma_format_s16;
    config.capture.channels = 1;
    config.playback.format = ma_format_s16;
    config.playback.channels = 1;
    config.dataCallback = Impl::callback;
    config.pUserData = impl.get();

    if (ma_device_init(nullptr, &config, &impl->device) != MA_SUCCESS) {
        JST_ERROR("[VOICE] Could not open the microphone and speaker.");
        return Result::ERROR;
    }
    impl->initialized = true;

    if (ma_device_start(&impl->device) != MA_SUCCESS) {
        stop();
        JST_ERROR("[VOICE] Could not start the microphone and speaker.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

void Session::Audio::stop() {
    if (!impl->initialized) {
        return;
    }

    ma_device_uninit(&impl->device);
    impl->initialized = false;
    impl->input.store(0.0f);
    impl->output.store(0.0f);

    {
        std::lock_guard<std::mutex> lock(impl->captureMutex);
        impl->captureQueue.clear();
    }
    flush();
}

std::vector<I16> Session::Audio::capture(U64 minimumSamples) {
    std::vector<I16> samples;

    std::lock_guard<std::mutex> lock(impl->captureMutex);
    if (impl->captureQueue.size() >= minimumSamples) {
        samples.swap(impl->captureQueue);
    }

    return samples;
}

void Session::Audio::flush() {
    std::lock_guard<std::mutex> lock(impl->playbackMutex);
    impl->playbackQueue.clear();
    impl->playbackOffset = 0;
}

void Session::Audio::play(const std::string& base64) {
    const auto bytes = Base64::Decode(base64);
    const U64 sampleCount = bytes.size() / sizeof(I16);
    if (sampleCount == 0) {
        return;
    }

    std::lock_guard<std::mutex> lock(impl->playbackMutex);
    auto& queue = impl->playbackQueue;

    const bool mostlyPlayed = impl->playbackOffset > 0 && impl->playbackOffset * 2 > queue.size();
    if (mostlyPlayed) {
        queue.erase(queue.begin(), queue.begin() + static_cast<std::ptrdiff_t>(impl->playbackOffset));
        impl->playbackOffset = 0;
    }

    const U64 maxSamples = static_cast<U64>(impl->sampleRate) * kMaxPlaybackSeconds;
    const U64 queued = queue.size() - impl->playbackOffset;
    if (queued + sampleCount > maxSamples) {
        return;
    }

    const U64 previous = impl->playbackQueue.size();
    impl->playbackQueue.resize(previous + sampleCount);
    std::memcpy(impl->playbackQueue.data() + previous, bytes.data(), sampleCount * sizeof(I16));
}

F32 Session::Audio::input() const {
    return impl->input.load(std::memory_order_relaxed);
}

F32 Session::Audio::output() const {
    return impl->output.load(std::memory_order_relaxed);
}

#else

struct Session::Audio::Impl {};

Session::Audio::Audio(U32) : impl(std::make_unique<Impl>()) {}

Session::Audio::~Audio() = default;

Result Session::Audio::start() {
    JST_ERROR("[VOICE] Built without audio support.");
    return Result::ERROR;
}

void Session::Audio::stop() {}

std::vector<I16> Session::Audio::capture(U64) {
    return {};
}

void Session::Audio::play(const std::string&) {}

void Session::Audio::flush() {}

F32 Session::Audio::input() const {
    return 0.0f;
}

F32 Session::Audio::output() const {
    return 0.0f;
}

#endif

}  // namespace Jetstream::Agent
