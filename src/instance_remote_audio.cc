#include "instance_remote_audio.hh"

#include "jetstream/logger.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <system_error>
#include <vector>

#include <opus.h>

namespace Jetstream {

namespace {

using Audio = Instance::Remote::Audio;

constexpr U64 kPrimeFrames = Audio::FrameSize * 2;
constexpr U64 kDriftLimitFrames = kPrimeFrames + Audio::FrameSize * 2;
constexpr U64 kDriftFloorFrames = kPrimeFrames - Audio::FrameSize / 2;
constexpr U64 kDriftTrimFrames = Audio::SampleRate / 1000;
constexpr U64 kLatencyLimitFrames = Audio::SampleRate / 5;
constexpr U64 kSourceCapacityFrames = Audio::SampleRate / 2;
constexpr U64 kPushChunkFrames = 512;
constexpr U64 kMaximumPacketBytes = 1500;
constexpr opus_int32 kBitrate = 96000;

struct Mixer {
    std::mutex mutex;
    std::vector<std::weak_ptr<Audio::Source>> sources;
    std::atomic<bool> active = false;
    bool owned = false;
};

Mixer& GetMixer() {
    static auto* mixer = new Mixer();
    return *mixer;
}

}  // namespace

struct Instance::Remote::Audio::Encoder {
    ~Encoder() {
        if (state) {
            opus_encoder_destroy(state);
        }
    }

    OpusEncoder* state = nullptr;
};

Instance::Remote::Audio::Source::Source() : buffer(kSourceCapacityFrames * Channels) {}

Instance::Remote::Audio::Audio() = default;

Instance::Remote::Audio::~Audio() {
    stop();
}

bool Instance::Remote::Audio::Active() {
    return GetMixer().active.load(std::memory_order_relaxed);
}

std::shared_ptr<Instance::Remote::Audio::Source> Instance::Remote::Audio::Attach() {
    auto source = std::make_shared<Source>();
    auto& mixer = GetMixer();
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

Result Instance::Remote::Audio::start(const std::chrono::steady_clock::time_point epoch, Sink output) {
    stop();

    auto& mixer = GetMixer();
    {
        std::lock_guard<std::mutex> lock(mixer.mutex);
        if (mixer.owned) {
            JST_WARN("[REMOTE] Another remote stream is already sending audio.");
            return Result::SUCCESS;
        }
    }

    int error = OPUS_OK;
    auto opened = std::make_unique<Encoder>();
    opened->state = opus_encoder_create(SampleRate, Channels, OPUS_APPLICATION_RESTRICTED_LOWDELAY, &error);
    if (error != OPUS_OK || !opened->state) {
        JST_ERROR("[REMOTE] Failed to create Opus encoder: {}", opus_strerror(error));
        return Result::ERROR;
    }
    if (opus_encoder_ctl(opened->state, OPUS_SET_BITRATE(kBitrate)) != OPUS_OK) {
        JST_ERROR("[REMOTE] Failed to set the Opus bitrate.");
        return Result::ERROR;
    }

    {
        std::lock_guard<std::mutex> lock(mixer.mutex);
        if (mixer.owned) {
            JST_WARN("[REMOTE] Another remote stream is already sending audio.");
            return Result::SUCCESS;
        }
        for (const auto& entry : mixer.sources) {
            if (const auto source = entry.lock()) {
                (void)source->buffer.clear();
                source->primed = false;
            }
        }
        mixer.owned = true;
        mixer.active = true;
    }

    encoder = std::move(opened);
    sink = std::move(output);
    {
        std::lock_guard<std::mutex> lock(mutex);
        running = true;
    }
    try {
        thread = std::thread([this, epoch]() {
            loop(epoch);
        });
    } catch (const std::system_error& e) {
        JST_ERROR("[REMOTE] Failed to start the audio thread: {}", e.what());
        stop();
        return Result::ERROR;
    }

    JST_DEBUG("[REMOTE] Audio stream started.");
    return Result::SUCCESS;
}

void Instance::Remote::Audio::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex);
        running = false;
    }
    condition.notify_all();
    if (thread.joinable()) {
        thread.join();
    }

    if (encoder) {
        auto& mixer = GetMixer();
        std::lock_guard<std::mutex> lock(mixer.mutex);
        mixer.active = false;
        mixer.owned = false;
    }
    encoder.reset();
    sink = {};
}

void Instance::Remote::Audio::loop(const std::chrono::steady_clock::time_point epoch) {
    std::vector<F32> mix(FrameSize * Channels);
    std::vector<F32> scratch(FrameSize * Channels);
    std::vector<U8> packet(kMaximumPacketBytes);
    std::vector<std::shared_ptr<Source>> sources;

    U64 tick = static_cast<U64>((std::chrono::steady_clock::now() - epoch) / FrameDuration) + 1;

    std::unique_lock<std::mutex> lock(mutex);
    while (running) {
        const auto deadline = epoch + tick * FrameDuration;
        if (condition.wait_until(lock, deadline, [&]() { return !running; })) {
            break;
        }
        lock.unlock();

        const auto now = std::chrono::steady_clock::now();
        if (now - deadline > FrameDuration * 5) {
            tick = static_cast<U64>((now - epoch) / FrameDuration);
        }

        sources.clear();
        {
            auto& mixer = GetMixer();
            std::lock_guard<std::mutex> mixerLock(mixer.mutex);
            for (const auto& entry : mixer.sources) {
                if (auto source = entry.lock()) {
                    sources.push_back(std::move(source));
                }
            }
        }

        std::fill(mix.begin(), mix.end(), 0.0f);
        bool mixed = false;
        for (const auto& source : sources) {
            U64 available = source->buffer.size() / Channels;
            if (!source->primed) {
                if (available < kPrimeFrames) {
                    continue;
                }
                source->primed = true;
            }
            if (available > kLatencyLimitFrames) {
                (void)source->buffer.discard((available - kPrimeFrames) * Channels);
                available = kPrimeFrames;
            } else if (available > kDriftLimitFrames) {
                (void)source->buffer.discard(kDriftTrimFrames * Channels);
                available -= kDriftTrimFrames;
            }

            const bool stretch = available >= FrameSize && available < kDriftFloorFrames;
            const U64 wanted = stretch ? FrameSize - kDriftTrimFrames : FrameSize;
            const U64 count = std::min(available, wanted);
            if (count < wanted) {
                source->primed = false;
            }
            if (count == 0 || source->buffer.pop(scratch.data(), count * Channels) != Result::SUCCESS) {
                continue;
            }
            for (U64 i = 0; i < count * Channels; ++i) {
                mix[i] += scratch[i];
            }
            if (stretch) {
                for (U64 i = count; i < FrameSize; ++i) {
                    for (U64 channel = 0; channel < Channels; ++channel) {
                        mix[i * Channels + channel] += scratch[(count - 1) * Channels + channel];
                    }
                }
            }
            mixed = true;
        }

        sources.clear();

        if (mixed) {
            for (auto& sample : mix) {
                sample = std::clamp(sample, -1.0f, 1.0f);
            }
            const opus_int32 bytes = opus_encode_float(encoder->state,
                                                       mix.data(),
                                                       static_cast<int>(FrameSize),
                                                       packet.data(),
                                                       static_cast<opus_int32>(packet.size()));
            if (bytes > 0) {
                sink(packet.data(), static_cast<U64>(bytes), tick * FrameSize);
            } else if (bytes < 0) {
                JST_DEBUG("[REMOTE] Failed to encode audio frame: {}", opus_strerror(bytes));
            }
        }

        ++tick;
        lock.lock();
    }
}

}  // namespace Jetstream
