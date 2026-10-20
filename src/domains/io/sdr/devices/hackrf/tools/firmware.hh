#ifndef JETSTREAM_DOMAINS_IO_SDR_DEVICES_HACKRF_TOOLS_FIRMWARE_HH
#define JETSTREAM_DOMAINS_IO_SDR_DEVICES_HACKRF_TOOLS_FIRMWARE_HH

#include <optional>
#include <vector>

#include "../../device.hh"

namespace Jetstream::Radio {

class JETSTREAM_API HackRfFirmware {
 public:
    static constexpr F64 SampleRate = 20.0e6;
    static constexpr F64 StepWidth = 20.0e6;
    static constexpr F64 Offset = 7.5e6;
    static constexpr F64 Granularity = 1.0e6;
    static constexpr U64 BlockBytes = 16384;
    static constexpr U64 MaxCaptureLength = 8180;
    static constexpr U64 MaxDwell = 64;
    static constexpr U64 MaxRangeEnd = 6000;

    struct Range {
        U64 start = 0;
        U64 end = 0;
    };

    static Device::Firmware Capability();
    static std::optional<Range> Snap(F64 startFrequency, F64 endFrequency);
    static std::vector<U64> Headers(const Range& range);
    static bool Layout(F64 startFrequency, F64 endFrequency, U64 captureLength,
                       Device::Layout& layout);
};

}  // namespace Jetstream::Radio

#endif  // JETSTREAM_DOMAINS_IO_SDR_DEVICES_HACKRF_TOOLS_FIRMWARE_HH
