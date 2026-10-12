#ifndef JETSTREAM_DOMAINS_IO_SDR_TOOLS_CAPTURE_QUEUE_HH
#define JETSTREAM_DOMAINS_IO_SDR_TOOLS_CAPTURE_QUEUE_HH

#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <span>
#include <string>
#include <vector>

#include "../devices/device.hh"

namespace Jetstream::Radio {

// Bounded sample queue between a vendor receive callback and `Device::read()`.
// Used by the RTL-SDR, Airspy, and HackRF adapters. `retune()` relabels a
// running stream: it drops every queued sample, the next `transfers` vendor
// callbacks, which can still hold samples from before the change, and then
// `discard` settling samples.

class JETSTREAM_API CaptureQueue {
 public:
    using Converter = std::function<void(CF32* output, U64 offset, U64 count)>;

    Result resize(U64 capacity);
    void reset(U64 generation, U64 discard, F64 frequency, F64 sampleRate);
    void retune(U64 generation, U64 transfers, U64 discard, F64 frequency);
    void close();
    void push(U64 generation, U64 count, const Converter& convert, U64 deviceLost = 0);
    void fail(Device::Status status, const std::string& error);
    Device::Capture pop(std::span<CF32> samples, std::chrono::milliseconds timeout);

 private:
    struct Gap {
        U64 position = 0;
        U64 lost = 0;
    };

    std::mutex mutex;
    std::condition_variable available;
    std::vector<CF32> storage;
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
