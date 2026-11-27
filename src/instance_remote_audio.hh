#pragma once

#include "jetstream/instance_remote.hh"
#include "jetstream/tools/circular_buffer.hh"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace Jetstream {

struct Instance::Remote::Audio {
    static constexpr U32 SampleRate = 48000;
    static constexpr U32 Channels = 2;
    static constexpr U64 FrameSize = SampleRate / 50;
    static constexpr std::chrono::milliseconds FrameDuration{20};

    struct Source {
        Source();

        Tools::CircularBuffer<F32> buffer;
        bool primed = false;
    };

    using Sink = std::function<void(const U8* data, U64 size, U64 sampleTime)>;

    static bool Active();
    static std::shared_ptr<Source> Attach();
    static void Push(Source& source, const F32* frames, U64 frameCount, U32 channels, F32 gain);

    Audio();
    ~Audio();

    Result start(std::chrono::steady_clock::time_point epoch, Sink sink);
    void stop();

 private:
    struct Encoder;

    struct Mixer {
        std::mutex mutex;
        std::vector<std::weak_ptr<Source>> sources;
        std::atomic<bool> active = false;
        bool owned = false;
    };

    static Mixer& Shared();

    void loop(std::chrono::steady_clock::time_point epoch);

    std::unique_ptr<Encoder> encoder;
    Sink sink;
    std::thread thread;
    std::mutex mutex;
    std::condition_variable condition;
    bool running = false;
};

}  // namespace Jetstream
