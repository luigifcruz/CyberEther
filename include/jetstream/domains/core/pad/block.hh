#ifndef JETSTREAM_DOMAINS_CORE_PAD_BLOCK_HH
#define JETSTREAM_DOMAINS_CORE_PAD_BLOCK_HH

#include "jetstream/block.hh"
#include "jetstream/types.hh"

namespace Jetstream::Blocks {

struct Pad : public Block::Config {
    U64 size = 0;
    U64 axis = 0;

    JST_BLOCK_TYPE(pad);
    JST_BLOCK_DOMAIN("Core");
    JST_BLOCK_PARAMS(size, axis);
    JST_BLOCK_DESCRIPTION(
        "Pad",
        "Adds zeros to the end of a tensor.",
        "# Pad\n"
        "The Pad block adds zero padding to the end of a tensor along a specified axis. "
        "This is commonly used to prepare data for operations that require specific sizes, "
        "such as FFT (power-of-2 sizes) or convolutions.\n\n"

        "## Arguments\n"
        "- **Pad Size**: Number of zeros to append to the specified axis.\n"
        "- **Pad Axis**: The dimension along which to add padding (0-indexed).\n\n"

        "## Useful For\n"
        "- Zero-padding for FFT operations to increase frequency resolution.\n"
        "- Aligning tensor sizes for batch processing.\n"
        "- Preparing data for overlap-add or overlap-save convolution.\n"
        "- Buffer extension for signal processing algorithms.\n\n"

        "## Examples\n"
        "- Pad 1D signal for FFT:\n"
        "  Input: F32[1000], Config: size=24, axis=0\n"
        "  Output: F32[1024] (zeros appended at the end)\n"
        "- Pad 2D tensor along rows:\n"
        "  Input: CF32[100, 256], Config: size=28, axis=0\n"
        "  Output: CF32[128, 256]\n\n"

        "## Implementation\n"
        "Input -> Copy to Output -> Fill Padding with Zeros -> Output\n"
        "The original data is copied to the beginning of the output tensor, "
        "and the remaining elements are filled with zeros.";
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_CORE_PAD_BLOCK_HH
