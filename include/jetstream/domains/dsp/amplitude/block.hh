#ifndef JETSTREAM_DOMAINS_DSP_AMPLITUDE_BLOCK_HH
#define JETSTREAM_DOMAINS_DSP_AMPLITUDE_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct Amplitude : public Block::Config {
    JST_BLOCK_TYPE(amplitude);
    JST_BLOCK_DOMAIN("DSP");
    JST_BLOCK_PARAMS();
    JST_BLOCK_DESCRIPTION(
        "Amplitude",
        "Calculates the amplitude of a signal in decibels.",
        "# Amplitude\n"
        "The Amplitude block computes the magnitude of complex or real signals and converts "
        "the result to decibels (dB). For complex inputs, it calculates sqrt(real^2 + imag^2). "
        "For real inputs, it takes the absolute value. The output is normalized using a "
        "scaling coefficient based on the signal length.\n\n"

        "## Useful For\n"
        "- Spectrum visualization and analysis.\n"
        "- Power spectrum display.\n"
        "- Signal level monitoring.\n"
        "- Audio and RF signal analysis.\n\n"

        "## Examples\n"
        "- Complex amplitude:\n"
        "  Input: CF32[1024] -> Output: F32[1024] (dB values)\n"
        "- Real amplitude:\n"
        "  Input: F32[1024] -> Output: F32[1024] (dB values)\n\n"

        "## Implementation\n"
        "Input Signal -> Magnitude Calculation -> dB Conversion -> Output\n"
        "1. For complex signals: magnitude = sqrt(I^2 + Q^2)\n"
        "2. For real signals: magnitude = |value|\n"
        "3. Convert to dB: 20 * log10(magnitude) + scaling_coefficient"
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_DSP_AMPLITUDE_BLOCK_HH
