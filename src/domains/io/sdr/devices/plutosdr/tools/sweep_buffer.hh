#ifndef JETSTREAM_DOMAINS_IO_SDR_DEVICES_PLUTOSDR_TOOLS_SWEEP_BUFFER_HH
#define JETSTREAM_DOMAINS_IO_SDR_DEVICES_PLUTOSDR_TOOLS_SWEEP_BUFFER_HH

#include <optional>

#include "jetstream/macros.hh"
#include "jetstream/types.hh"

namespace Jetstream::Radio {

struct JETSTREAM_API PlutoSweepBufferLayout {
    U64 captureSamples = 0;
    U64 batchSamples = 0;
    U64 transferSamples = 0;

    static std::optional<PlutoSweepBufferLayout> Create(U64 captureLength, U64 dwell);
};

}

#endif
