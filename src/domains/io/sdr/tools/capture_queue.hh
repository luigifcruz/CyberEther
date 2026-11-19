#ifndef JETSTREAM_DOMAINS_IO_SDR_TOOLS_CAPTURE_QUEUE_HH
#define JETSTREAM_DOMAINS_IO_SDR_TOOLS_CAPTURE_QUEUE_HH

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <mutex>
#include <span>
#include <string>
#include <vector>

#include "../devices/device.hh"

namespace Jetstream::Radio {

class JETSTREAM_API CaptureQueue {
 public:
    using Converter = std::function<void(CF32* output, U64 offset, U64 count)>;

    Result resize(U64 capacity, U64 sampleBytes = sizeof(CF32));
    void reset(U64 generation, U64 discard, F64 frequency, F64 sampleRate);
    void retune(U64 generation, U64 transfers, U64 discard, F64 frequency);
    void close();
    void push(U64 generation, U64 count, const Converter& convert, U64 deviceLost = 0);
    void push(U64 generation, U64 count, const std::byte* samples, U64 deviceLost = 0);
    void fail(Device::Status status, const std::string& error);
    Device::Capture pop(std::span<CF32> samples, std::chrono::milliseconds timeout);
    Device::Capture pop(std::span<std::byte> samples, std::chrono::milliseconds timeout);

 private:
    struct Gap {
        U64 position = 0;
        U64 lost = 0;
    };

    template <typename Copy>
    void write(U64 generation, U64 count, U64 deviceLost, const Copy& copy);

    std::mutex mutex;
    std::condition_variable available;
    std::vector<std::byte> storage;
    U64 sampleBytes = sizeof(CF32);
    U64 head = 0;
    U64 occupancy = 0;
    U64 pushed = 0;
    U64 popped = 0;
    std::deque<Gap> gaps;
    U64 generation = 0;
    U64 discardTransfers = 0;
    U64 discard = 0;
    bool open = false;
    Device::Status failure = Device::Status::Samples;
    std::string failureMessage;
    F64 frequency = 0.0;
    F64 sampleRate = 0.0;
};

}  // namespace Jetstream::Radio

#endif  // JETSTREAM_DOMAINS_IO_SDR_TOOLS_CAPTURE_QUEUE_HH
