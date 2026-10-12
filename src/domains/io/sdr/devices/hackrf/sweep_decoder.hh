#ifndef JETSTREAM_DOMAINS_IO_SDR_DEVICES_HACKRF_SWEEP_DECODER_HH
#define JETSTREAM_DOMAINS_IO_SDR_DEVICES_HACKRF_SWEEP_DECODER_HH

#include <optional>
#include <vector>

#include "../device.hh"
#include "transfer_queue.hh"

namespace Jetstream::Radio {

class JETSTREAM_API SweepDecoder {
 public:
    static constexpr U64 HeaderBytes = 10;

    struct Block {
        U64 frequency = 0;
        const I8* payload = nullptr;
        U64 payloadSamples = 0;

        static std::optional<Block> Parse(const U8* data, U64 size, U64 blockBytes);
    };

    class Sequence {
     public:
        enum class Event : U8 {
            Waiting,
            Capture,
            Invalid,
        };

        void configure(std::vector<U64> frequencies, U64 dwell);
        void invalidate();
        Event next(U64 frequency, U64& hop, U64& repetition);
        U64 losses() const;

     private:
        std::vector<U64> frequencies;
        U64 dwell = 1;
        bool synchronized = false;
        bool seenOther = false;
        U64 hop = 0;
        U64 repetition = 0;
        U64 lost = 0;
    };

    void configure(std::vector<U64> headers, U64 dwell, U64 captureLength, U64 blockBytes,
                   F64 offset);
    bool decode(const TransferQueue::Transfer& transfer, const Device::SweepSink& sink);
    U64 blocks() const;
    U64 losses() const;

 private:
    Sequence sequence;
    std::vector<CF32> captures;
    U64 dwell = 1;
    U64 captureLength = 0;
    U64 blockBytes = 0;
    F64 offset = 0.0;
    U64 valid = 0;
};

}  // namespace Jetstream::Radio

#endif  // JETSTREAM_DOMAINS_IO_SDR_DEVICES_HACKRF_SWEEP_DECODER_HH
