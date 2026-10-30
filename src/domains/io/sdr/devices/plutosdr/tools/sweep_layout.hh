#ifndef JETSTREAM_DOMAINS_IO_SDR_DEVICES_PLUTOSDR_TOOLS_SWEEP_LAYOUT_HH
#define JETSTREAM_DOMAINS_IO_SDR_DEVICES_PLUTOSDR_TOOLS_SWEEP_LAYOUT_HH

#include <optional>

#include "jetstream/macros.hh"
#include "jetstream/types.hh"

namespace Jetstream::Radio {

struct JETSTREAM_API PlutoSdrSweepLayout {
    U64 captureSamples = 0;
    U64 batchSamples = 0;
    U64 transferSamples = 0;

    static std::optional<PlutoSdrSweepLayout> Create(U64 captureLength, U64 dwell);
};

}  // namespace Jetstream::Radio

#endif  // JETSTREAM_DOMAINS_IO_SDR_DEVICES_PLUTOSDR_TOOLS_SWEEP_LAYOUT_HH
