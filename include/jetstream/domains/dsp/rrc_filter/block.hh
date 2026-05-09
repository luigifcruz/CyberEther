#ifndef JETSTREAM_DOMAINS_DSP_RRC_FILTER_BLOCK_HH
#define JETSTREAM_DOMAINS_DSP_RRC_FILTER_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct RrcFilter : public Block::Config {
    F32 symbolRate = 1.0e6f;
    F32 sampleRate = 2.0e6f;
    F32 rollOff = 0.35f;
    U64 taps = 101;

    JST_BLOCK_TYPE(rrc_filter);
    JST_BLOCK_DOMAIN("DSP");
    JST_BLOCK_PARAMS(symbolRate, sampleRate, rollOff, taps);
    JST_BLOCK_DESCRIPTION(
        "RRC Filter",
        "Root raised cosine matched filter for PSK modulation.",
        "# RRC Filter\n"
        "The RRC Filter block implements a root raised cosine (RRC) "
        "matched filter optimized for PSK (Phase Shift Keying) "
        "modulation schemes including BPSK, QPSK, and 8PSK. This "
        "filter provides optimal signal-to-noise ratio performance "
        "for digital communications by matching the transmitter's "
        "pulse shaping filter.\n\n"

        "## Arguments\n"
        "- **Sample Rate**: Sampling rate of the input signal.\n"
        "- **Symbol Rate**: Symbol rate of the modulated signal.\n"
        "- **Roll-off Factor**: Bandwidth-efficiency trade-off "
        "(0.0 to 1.0).\n"
        "- **Taps**: Number of filter coefficients (must be odd "
        "and >= 3).\n\n"

        "## Useful For\n"
        "- Matched filtering in PSK demodulation systems.\n"
        "- Maximizing signal-to-noise ratio in digital receivers.\n"
        "- Symbol timing recovery and synchronization.\n"
        "- Reducing intersymbol interference (ISI).\n\n"

        "## Examples\n"
        "- QPSK demodulation:\n"
        "  Config: Symbol Rate=1 MHz, Sample Rate=4 MHz, "
        "Roll-off=0.35\n"
        "  Input: CF32[8192] -> Output: CF32[8192]\n\n"

        "## Implementation\n"
        "Input -> RRC FIR Convolution -> Output\n"
        "1. Compute RRC pulse shape coefficients.\n"
        "2. Convolve input with coefficients using a circular "
        "history buffer.\n"
        "3. Output has same shape and type as input."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_DSP_RRC_FILTER_BLOCK_HH
