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

        "The input must be F32 or CF32, provide `sampleAxis` metadata, and may "
        "provide `batchAxis`. Real samples in F32 format are promoted to CF32 "
        "with a zero imaginary component. The filter output is always CF32. "
        "Supports multiple center frequencies (heads) to extract multiple "
        "channels simultaneously from a wideband capture. The generated head "
        "dimension is published as `channelAxis`. Inputs that already have a "
        "channel role are rejected.\n\n"

        "When the resampler ratio (sample rate / bandwidth) is an integer and "
        "the signal and overlap extents are divisible by that ratio, the output "
        "will be automatically resampled to the filter bandwidth.\n\n"

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
        "  Input: CF32[8190] -> Output: CF32[1, 819]\n"
        "- Real input promotion:\n"
        "  Input: F32[8190] -> Output: CF32[1, 819]\n"
        "- Multi-head extraction:\n"
        "  Config: Sample Rate=2 MHz, Bandwidth=0.2 MHz, Heads=3, Center=[0, 0.4, -0.4]\n"
        "  Input: CF32[8190] -> Output: CF32[3, 819]\n\n"

        "## Implementation\n"
        "Cast to CF32 -> FilterTaps -> Pad -> FFT -> Multiply -> (Fold) -> IFFT -> "
        "(Unpad -> Overlap-Add) -> Output\n"
        "1. Generates CF32 FIR filter coefficients and promotes real input to CF32.\n"
        "2. Pads both operands and transforms them into full-length CF32 spectra.\n"
        "3. Multiplies corresponding bins using complex multiplication. Packed F32 "
        "spectra are never passed to generic arithmetic.\n"
        "4. Optionally folds (decimates) for resampling.\n"
        "5. Inverse FFTs and normalizes once. Multi-tap filters remove padding and "
        "apply overlap-add for continuity."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_DSP_FILTER_BLOCK_HH
