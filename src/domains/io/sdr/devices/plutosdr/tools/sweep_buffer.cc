#include "sweep_buffer.hh"

#include <algorithm>
#include <cstddef>
#include <limits>

namespace Jetstream::Radio {

std::optional<PlutoSweepBufferLayout> PlutoSweepBufferLayout::Create(const U64 captureLength,
                                                                     const U64 dwell) {
    constexpr U64 Maximum = 1 << 20;
    constexpr U64 Minimum = 1024;
    constexpr U64 Preferred = 8192;
    constexpr U64 Alignment = 16;

    if (captureLength == 0 || captureLength > Maximum || dwell == 0 ||
        dwell > (std::numeric_limits<std::size_t>::max() / sizeof(CF32)) / captureLength) {
        return std::nullopt;
    }
    const U64 batch = captureLength * std::min(dwell, std::max<U64>(1, Preferred / captureLength));
    return PlutoSweepBufferLayout{
        .captureSamples = captureLength * dwell,
        .batchSamples = batch,
        .transferSamples = std::max(Minimum, ((batch + Alignment - 1) / Alignment) * Alignment),
    };
}

}
