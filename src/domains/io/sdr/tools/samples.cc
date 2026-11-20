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

void ConvertI16Channels(const I16* input, std::span<CF32* const> outputs, const U64 count,
                        const F32 scale) {
    const U64 channels = outputs.size();
    for (U64 i = 0; i < count; ++i) {
        for (U64 channel = 0; channel < channels; ++channel) {
            const I16* word = input + 2 * (i * channels + channel);
            outputs[channel][i] = {static_cast<F32>(word[0]) / scale,
                                   static_cast<F32>(word[1]) / scale};
        }
    }
}

void SplitI16Channels(const I16* input, std::span<I16* const> outputs, const U64 count) {
    const U64 channels = outputs.size();
    for (U64 i = 0; i < count; ++i) {
        for (U64 channel = 0; channel < channels; ++channel) {
            const I16* word = input + 2 * (i * channels + channel);
            outputs[channel][2 * i] = word[0];
            outputs[channel][2 * i + 1] = word[1];
        }
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
