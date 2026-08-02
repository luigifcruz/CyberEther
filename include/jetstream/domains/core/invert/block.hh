#ifndef JETSTREAM_DOMAINS_CORE_INVERT_BLOCK_HH
#define JETSTREAM_DOMAINS_CORE_INVERT_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct Invert : public Block::Config {
    JST_BLOCK_TYPE(invert);
    JST_BLOCK_DOMAIN("Core");
    JST_BLOCK_NODE_SIZE(XS);
    JST_BLOCK_PARAMS();
    JST_BLOCK_DESCRIPTION(
        "Invert",
        "Time-domain modulation for FFT shift.",
        "# Invert\n"
        "The Invert block modulates the input signal to center a subsequent FFT. "
        "Even-length signals use [1, -1, 1, -1, ...]. Odd-length signals use the "
        "equivalent integer-bin complex phasor to avoid a half-bin shift. The pattern "
        "runs along the dimension "
        "identified by `sampleAxis` and restarts for every combination of the remaining "
        "dimensions. Optional `batchAxis` and `channelAxis` metadata, along with all other "
        "axis metadata, are preserved.\n\n"

        "## Useful For\n"
        "- Centering FFT output for spectrum visualization.\n"
        "- Performing frequency domain shifts.\n"
        "- Pre-processing signals before spectral analysis.\n\n"

        "## Examples\n"
        "- Complex signal inversion:\n"
        "  Input: CF32[1024] -> Output: CF32[1024]\n"
        "  [c0, c1, c2, c3, ...] -> [c0, -c1, c2, -c3, ...]\n\n"

        "## Implementation\n"
        "Input -> FFT Shift Modulation -> Output\n"
        "1. Input signal is processed element by element.\n"
        "2. Even lengths use alternating sign inversion.\n"
        "3. Odd lengths use an exact integer-bin complex phasor.\n"
        "4. Output has the same shape as input.";
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_CORE_INVERT_BLOCK_HH
