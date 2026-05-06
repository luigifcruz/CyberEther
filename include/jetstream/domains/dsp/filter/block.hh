#ifndef JETSTREAM_DOMAINS_DSP_FILTER_BLOCK_HH
#define JETSTREAM_DOMAINS_DSP_FILTER_BLOCK_HH

#include <vector>

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct Filter : public Block::Config {
    F32 sampleRate = 2.0e6f;
    F32 bandwidth = 1.0e6f;
    std::vector<F32> center = {0.0e6f};
    U64 taps = 101;
    U64 heads = 1;

    JST_BLOCK_TYPE(filter);
    JST_BLOCK_DOMAIN("DSP");
    JST_BLOCK_PARAMS(sampleRate, bandwidth, center, taps, heads);
    JST_BLOCK_DESCRIPTION(
        "Filter",
        "Filters input signal with a FIR bandpass filter.",
        "# Filter\n"
        "The Filter block generates FIR bandpass filter coefficients and applies "
        "them to the input signal using frequency-domain overlap-add convolution. "
        "It combines the Filter Taps and Filter Engine functionality into a single "
        "block for convenience.\n\n"

        "Supports multiple center frequencies (heads) to extract multiple "
        "channels simultaneously from a wideband capture.\n\n"

        "When the resampler ratio (sample rate / bandwidth) is an integer, the "
        "output will be automatically resampled to the filter bandwidth.\n\n"

        "## Arguments\n"
        "- **Sample Rate**: The sampling rate of the input signal in MHz.\n"
        "- **Bandwidth**: The passband bandwidth of the filter in MHz.\n"
        "- **Heads**: Number of parallel filter heads (one per center frequency).\n"
        "- **Center**: Center frequency offset(s) of the filter in MHz.\n"
        "- **Taps**: The number of filter coefficients (must be odd).\n\n"

        "## Useful For\n"
        "- One-step bandpass filtering and optional resampling.\n"
        "- Multi-channel FM station extraction from wideband captures.\n"
        "- Narrowband signal isolation.\n\n"

        "## Examples\n"
        "- Single-channel filter with resampling:\n"
        "  Config: Sample Rate=2 MHz, Bandwidth=0.2 MHz, Heads=1, Taps=101\n"
        "  Input: CF32[8192] -> Output: CF32[819]\n"
        "- Multi-head extraction:\n"
        "  Config: Sample Rate=2 MHz, Bandwidth=0.2 MHz, Heads=3, Center=[0, 0.4, -0.4]\n"
        "  Input: CF32[8192] -> Output: CF32[3, 819]\n\n"

        "## Implementation\n"
        "FilterTaps -> Pad -> FFT -> Multiply -> (Fold) -> IFFT -> Unpad -> "
        "Overlap-Add -> Output\n"
        "1. Generates FIR filter coefficients internally.\n"
        "2. Pads the signal and filter to the combined length.\n"
        "3. Multiplies spectra in the frequency domain.\n"
        "4. Optionally folds (decimates) for resampling.\n"
        "5. Inverse FFTs and applies overlap-add for continuity."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_DSP_FILTER_BLOCK_HH
