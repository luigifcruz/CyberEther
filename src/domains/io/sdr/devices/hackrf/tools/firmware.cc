#include "firmware.hh"

#include <algorithm>
#include <cmath>

namespace Jetstream::Radio {

Device::Firmware HackRfFirmware::Capability() {
    return {
        .sampleRate = SampleRate,
        .maxDwell = MaxDwell,
        .maxCaptureLength = MaxCaptureLength,
        .captureBase = 4,
        .captureStep = 8,
        .layout = &Layout,
    };
}

std::optional<HackRfFirmware::Range> HackRfFirmware::Snap(const F64 startFrequency,
                                                          const F64 endFrequency) {
    const auto step = static_cast<U64>(StepWidth / Granularity);
    const F64 start = std::floor(startFrequency / Granularity);
    const F64 end = std::ceil(endFrequency / Granularity - 1e-9);
    if (!std::isfinite(start) || !std::isfinite(end) || start < 0.0 || end < 0.0 ||
        start >= static_cast<F64>(MaxRangeEnd) || end > static_cast<F64>(MaxRangeEnd)) {
        return std::nullopt;
    }
    const auto low = static_cast<U64>(start);
    const auto high = std::max(static_cast<U64>(end), low);
    const U64 steps = std::max<U64>(1, (high - low + step - 1) / step);
    if (steps > (MaxRangeEnd - low) / step) {
        return std::nullopt;
    }
    return Range{.start = low, .end = low + steps * step};
}

std::vector<U64> HackRfFirmware::Headers(const Range& range) {
    const auto granularity = static_cast<U64>(Granularity);
    const auto step = static_cast<U64>(StepWidth);
    std::vector<U64> result;
    if (range.end <= range.start) {
        return result;
    }
    U64 frequency = range.start * granularity;
    bool odd = true;
    while (result.size() < 100000) {
        result.push_back(frequency);
        if (!odd && frequency + step >= range.end * granularity) {
            break;
        }
        frequency += odd ? step / 4 : 3 * step / 4;
        odd = !odd;
    }
    return result;
}

bool HackRfFirmware::Layout(const F64 startFrequency, const F64 endFrequency,
                            const U64 captureLength, Device::Layout& layout) {
    const auto range = Snap(startFrequency, endFrequency);
    if (!range || captureLength == 0) {
        return false;
    }

    layout = {};
    layout.startFrequency = static_cast<F64>(range->start) * Granularity;
    layout.endFrequency = static_cast<F64>(range->end) * Granularity;

    const U64 quarter = captureLength / 4;
    const U64 center = captureLength / 2;
    const U64 bands[2] = {1 + (captureLength * 5) / 8, 1 + captureLength / 8};
    for (const auto header : Headers(*range)) {
        Device::Layout::Hop hop{.frequency = static_cast<F64>(header) + Offset};
        for (const auto band : bands) {
            hop.bands.push_back({.offset = (band + center) % captureLength, .length = quarter});
        }
        layout.hops.push_back(std::move(hop));
    }
    return true;
}

}  // namespace Jetstream::Radio
