#include "samples.hh"

#include <cmath>
#include <limits>

namespace Jetstream::Radio {

void ConvertU8(const U8* input, CF32* output, const U64 count) {
    for (U64 i = 0; i < count; ++i) {
        output[i] = {(static_cast<F32>(input[2 * i]) - 127.5f) / 127.5f,
                     (static_cast<F32>(input[2 * i + 1]) - 127.5f) / 127.5f};
    }
}

void ConvertI8(const I8* input, CF32* output, const U64 count) {
    for (U64 i = 0; i < count; ++i) {
        output[i] = {static_cast<F32>(input[2 * i]) / 128.0f,
                     static_cast<F32>(input[2 * i + 1]) / 128.0f};
    }
}

void ConvertI16(const I16* input, CF32* output, const U64 count, const F32 scale) {
    for (U64 i = 0; i < count; ++i) {
        output[i] = {static_cast<F32>(input[2 * i]) / scale,
                     static_cast<F32>(input[2 * i + 1]) / scale};
    }
}

U64 GuardSamples(const F64 seconds, const F64 sampleRate) {
    if (!std::isfinite(seconds) || !std::isfinite(sampleRate) ||
        seconds < 0.0 || sampleRate <= 0.0) {
        return std::numeric_limits<U64>::max();
    }
    if (seconds == 0.0) {
        return 0;
    }
    const F64 samples = std::ceil(seconds * sampleRate - 1e-9);
    if (!std::isfinite(samples) ||
        samples >= static_cast<F64>(std::numeric_limits<U64>::max())) {
        return std::numeric_limits<U64>::max();
    }
    return static_cast<U64>(samples);
}

}  // namespace Jetstream::Radio
