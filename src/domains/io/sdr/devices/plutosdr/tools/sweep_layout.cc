#include "sweep_layout.hh"

#include <algorithm>
#include <cstddef>
#include <limits>

namespace Jetstream::Radio {

namespace {

constexpr U64 MaxCaptureSamples = 1 << 20;
constexpr U64 MinTransferSamples = 1024;
constexpr U64 PreferredBatchSamples = 8192;
constexpr U64 TransferAlignment = 16;

}  // namespace

std::optional<PlutoSdrSweepLayout> PlutoSdrSweepLayout::Create(const U64 captureLength,
                                                               const U64 dwell) {
    if (captureLength == 0 || captureLength > MaxCaptureSamples || dwell == 0 ||
        dwell > (std::numeric_limits<std::size_t>::max() / sizeof(CF32)) / captureLength) {
        return std::nullopt;
    }
    const U64 groups = std::max<U64>(1, PreferredBatchSamples / captureLength);
    const U64 batch = captureLength * std::min(dwell, groups);
    const U64 aligned = ((batch + TransferAlignment - 1) / TransferAlignment) * TransferAlignment;
    return PlutoSdrSweepLayout{
        .captureSamples = captureLength * dwell,
        .batchSamples = batch,
        .transferSamples = std::max(MinTransferSamples, aligned),
    };
}

}  // namespace Jetstream::Radio
