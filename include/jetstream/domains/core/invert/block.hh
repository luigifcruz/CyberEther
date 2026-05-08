#ifndef JETSTREAM_DOMAINS_CORE_INVERT_BLOCK_HH
#define JETSTREAM_DOMAINS_CORE_INVERT_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct Invert : public Block::Config {
    JST_BLOCK_TYPE(invert);
    JST_BLOCK_DOMAIN("Core");
    JST_BLOCK_PARAMS();
    JST_BLOCK_DESCRIPTION(
        "Invert",
        "Alternating sign inversion for FFT shift.",
        "# Invert\n"
        "The Invert block performs alternating sign inversion on the input signal, "
        "negating every other element. This operation is equivalent to multiplying "
        "the signal by [1, -1, 1, -1, ...] and is commonly used for FFT shift "
        "operations to center the spectrum.\n\n"

        "## Useful For\n"
        "- Centering FFT output for spectrum visualization.\n"
        "- Performing frequency domain shifts.\n"
        "- Pre-processing signals before spectral analysis.\n\n"

        "## Examples\n"
        "- Complex signal inversion:\n"
        "  Input: CF32[1024] -> Output: CF32[1024]\n"
        "  [c0, c1, c2, c3, ...] -> [c0, -c1, c2, -c3, ...]\n\n"

        "## Implementation\n"
        "Input -> Alternating Sign -> Output\n"
        "1. Input signal is processed element by element.\n"
        "2. Even-indexed elements are kept unchanged.\n"
        "3. Odd-indexed elements are negated.\n"
        "4. Output has the same shape as input.";
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_CORE_INVERT_BLOCK_HH
