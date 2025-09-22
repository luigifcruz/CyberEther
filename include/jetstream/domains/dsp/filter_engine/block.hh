#ifndef JETSTREAM_DOMAINS_DSP_FILTER_ENGINE_BLOCK_HH
#define JETSTREAM_DOMAINS_DSP_FILTER_ENGINE_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct FilterEngine : public Block::Config {
    JST_BLOCK_TYPE(filter_engine);
    JST_BLOCK_PARAMS();
    JST_BLOCK_DESCRIPTION(
        "Filter Engine",
        "Filters a signal using FIR filter coefficients.",
        "# Filter Engine\n"
        "The Filter Engine block applies a frequency-domain FIR filter to an "
        "input signal using the overlap-add method. It takes a signal and filter "
        "coefficients (typically from the Filter Taps block) as inputs.\n\n"

        "When the filter coefficients carry sample rate, bandwidth, and center "
        "frequency attributes, the engine will attempt to resample the output "
        "to match the filter bandwidth.\n\n"

        "## Useful For\n"
        "- Applying bandpass filters to wideband signals.\n"
        "- Frequency selection and channelization.\n"
        "- Decimation with filtering.\n\n"

        "## Examples\n"
        "- Apply filter taps to a signal:\n"
        "  Signal Input: CF32[8192], Filter Input: CF32[101]\n"
        "  Output: CF32[8192] (or resampled if bandwidth attributes are set)\n\n"

        "## Implementation\n"
        "Signal + Filter -> Pad -> FFT -> Multiply -> (Fold) -> IFFT -> "
        "Unpad -> Overlap-Add -> Output\n"
        "1. Pads the signal and filter to the combined length.\n"
        "2. Transforms both to frequency domain via FFT.\n"
        "3. Multiplies the spectra element-wise.\n"
        "4. Optionally folds (decimates) the spectrum for resampling.\n"
        "5. Inverse FFTs back to time domain.\n"
        "6. Removes padding and applies overlap-add for continuity."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_DSP_FILTER_ENGINE_BLOCK_HH
