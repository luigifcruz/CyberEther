#ifndef JETSTREAM_DOMAINS_DSP_FILTER_TAPS_BLOCK_HH
#define JETSTREAM_DOMAINS_DSP_FILTER_TAPS_BLOCK_HH

#include <vector>

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct FilterTaps : public Block::Config {
    F32 sampleRate = 2.0e6f;
    F32 bandwidth = 1.0e6f;
    std::vector<F32> center = {0.0e6f};
    U64 taps = 101;
    U64 heads = 1;

    JST_BLOCK_TYPE(filter_taps);
    JST_BLOCK_DOMAIN("DSP");
    JST_BLOCK_PARAMS(sampleRate, bandwidth, center, taps, heads);
    JST_BLOCK_DESCRIPTION(
        "Filter Taps",
        "Generates FIR bandpass filter coefficients.",
        "# Filter Taps\n"
        "The Filter Taps block creates Finite Impulse Response (FIR) bandpass "
        "filter coefficients (taps) based on specified frequency parameters. "
        "These taps can be used by a Filter Engine block to filter input data, "
        "attenuating or amplifying certain frequency components.\n\n"

        "Supports multiple center frequencies (heads) to generate multiple "
        "bandpass filters simultaneously. When multiple centers are specified, "
        "the output tensor is 2D with shape {heads, taps}.\n\n"

        "## Arguments\n"
        "- **Sample Rate**: The sampling rate of the signal in MHz.\n"
        "- **Bandwidth**: The passband bandwidth of the filter in MHz.\n"
        "- **Heads**: Number of filter heads (parallel center frequencies).\n"
        "- **Center**: Center frequency offset(s) of the filter in MHz.\n"
        "- **Taps**: The number of filter coefficients (must be odd).\n\n"

        "## Useful For\n"
        "- Creating bandpass filter coefficients for signal filtering.\n"
        "- Multi-channel frequency selection and channelization.\n"
        "- Narrowband signal extraction from wideband captures.\n\n"

        "## Examples\n"
        "- Single-head filter taps:\n"
        "  Config: Sample Rate=2 MHz, Bandwidth=0.2 MHz, Heads=1, Taps=101\n"
        "  Output: CF32[101]\n"
        "- Multi-head filter taps:\n"
        "  Config: Sample Rate=2 MHz, Bandwidth=0.2 MHz, Heads=3, Taps=101\n"
        "  Output: CF32[3, 101]\n\n"

        "## Implementation\n"
        "Config -> Sinc Function -> Blackman Window -> Upconversion -> Coefficients\n"
        "1. Generates a sinc function based on the bandwidth-to-sample-rate ratio.\n"
        "2. Applies a Blackman window to reduce spectral leakage.\n"
        "3. Upconverts each head to its specified center frequency.\n"
        "4. Outputs CF32 coefficients with sample_rate, bandwidth, and center "
        "attributes attached to the output tensor."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_DSP_FILTER_TAPS_BLOCK_HH
