#ifndef JETSTREAM_DOMAINS_IO_SDR_TOOLS_STREAM_BUFFER_HH
#define JETSTREAM_DOMAINS_IO_SDR_TOOLS_STREAM_BUFFER_HH

#include <chrono>
#include <mutex>

#include "jetstream/macros.hh"
#include "jetstream/tools/circular_buffer.hh"
#include "jetstream/types.hh"

namespace Jetstream::Radio {

class JETSTREAM_API StreamBuffer {
 public:
    struct Metadata {
        F64 frequency = 0.0;
        F64 sampleRate = 0.0;
        U64 generation = 0;

        bool operator==(const Metadata&) const = default;
    };

    struct Statistics {
        U64 pushedSamples = 0;
        U64 overwrittenSamples = 0;
        U64 rejectedSamples = 0;
    };

    Result resize(U64 capacity);
    void reset(const Metadata& metadata);
    bool push(U64 generation, const CF32* data, U64 count);
    Result pop(CF32* data, U64 count, Metadata& metadata);
    Result waitForSize(U64 count, std::chrono::milliseconds timeout);

    Metadata metadata() const;
    U64 size() const;
    U64 capacity() const;
    F64 throughput() const;
    Statistics statistics() const;

 private:
    mutable std::mutex mutex;
    Tools::CircularBuffer<CF32> samples;
    Metadata current;
    Statistics accumulated;
};

}  // namespace Jetstream::Radio

#endif  // JETSTREAM_DOMAINS_IO_SDR_TOOLS_STREAM_BUFFER_HH
