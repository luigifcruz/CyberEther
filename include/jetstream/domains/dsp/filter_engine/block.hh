#ifndef JETSTREAM_DOMAINS_DSP_FILTER_ENGINE_BLOCK_HH
#define JETSTREAM_DOMAINS_DSP_FILTER_ENGINE_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct FilterEngine : public Block::Config {
    JST_BLOCK_TYPE(filter_engine);
    JST_BLOCK_DOMAIN("DSP");
    JST_BLOCK_NODE_SIZE(XS);
    JST_BLOCK_PARAMS();
    JST_BLOCK_DESCRIPTION(
        "Filter Engine",
        "Filters a signal using FIR filter coefficients.",
        "# Filter Engine\n"
        "The Filter Engine block applies a frequency-domain FIR filter to an "
        "input signal using the overlap-add method. Signal inputs require explicit "
        "`sampleAxis` metadata. Filter coefficients must be rank-1 `[T]` with "
        "`sampleAxis=0`, or a rank-2 `[C,T]` bank with `channelAxis=0` and "
        "`sampleAxis=1`. Filter banks add a channel dimension immediately before "
        "the signal's sample dimension. Inputs may use F32 or CF32. Real values "
        "in F32 format are promoted to CF32 before filtering. The output always "
        "uses CF32.\n\n"

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
        "  Output: CF32[8192] (or resampled if bandwidth attributes are set)\n"
        "- Promote a real signal for complex taps:\n"
        "  Signal Input: F32[8192], Filter Input: CF32[101]\n"
        "  Output: CF32[8192]\n"
        "- Promote real signal and filter values:\n"
        "  Signal Input: F32[8192], Filter Input: F32[101]\n"
        "  Output: CF32[8192]\n"
        "- Apply three filter heads to a leading batch:\n"
        "  Signal Input: CF32[8, 8192], Filter Input: CF32[3, 101]\n"
        "  Output: CF32[8, 3, 8192]\n\n"

        "## Implementation\n"
        "Signal + Filter -> (Cast to CF32) -> Pad -> FFT -> Multiply -> (Fold) -> IFFT -> "
        "(Phase Correction) -> (Unpad -> Overlap-Add) -> Output\n"
        "1. Promotes both operands to CF32 before entering spectral arithmetic.\n"
        "2. Pads the signal and filter to the combined length.\n"
        "3. Transforms both into full-length CF32 spectra.\n"
        "4. Multiplies corresponding bins using complex multiplication. Packed F32 "
        "spectra are never passed to generic arithmetic.\n"
        "5. Optionally folds (decimates) the spectrum for resampling.\n"
        "6. Inverse FFTs, normalizes once, and preserves centered-resampling phase "
        "across blocks.\n"
        "7. Multi-tap filters remove padding and apply overlap-add for continuity."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_DSP_FILTER_ENGINE_BLOCK_HH
