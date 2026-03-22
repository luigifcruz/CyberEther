#ifndef JETSTREAM_DOMAINS_CORE_UNPAD_BLOCK_HH
#define JETSTREAM_DOMAINS_CORE_UNPAD_BLOCK_HH

#include "jetstream/block.hh"
#include "jetstream/types.hh"

namespace Jetstream::Blocks {

struct Unpad : public Block::Config {
    U64 size = 0;
    U64 axis = 0;

    JST_BLOCK_TYPE(unpad);
    JST_BLOCK_PARAMS(size, axis);
    JST_BLOCK_DESCRIPTION(
        "Unpad",
        "Removes padding from a tensor.",
        "# Unpad\n"
        "The Unpad block removes elements from the end of a tensor along a "
        "specified axis and exposes both the trimmed result and the removed "
        "portion as separate outputs. This is the inverse operation of the Pad "
        "block and is useful for stripping zero-padding after FFT or convolution.\n\n"

        "## Arguments\n"
        "- **Pad Size**: Number of elements to remove from the end of the specified axis.\n"
        "- **Pad Axis**: The dimension along which to remove padding (0-indexed).\n\n"

        "## Useful For\n"
        "- Removing zero-padding after FFT/IFFT operations.\n"
        "- Extracting valid convolution results from full output.\n"
        "- Implementing overlap-add algorithms (the pad output feeds the overlap input).\n"
        "- Restoring original tensor size after padded processing.\n\n"

        "## Examples\n"
        "- Remove FFT padding:\n"
        "  Config: Pad Size=24, Pad Axis=0\n"
        "  Input: F32[1024] -> Output: F32[1000], Pad: F32[24]\n"
        "- Remove padding from 2D tensor along rows:\n"
        "  Config: Pad Size=28, Pad Axis=0\n"
        "  Input: CF32[128, 256] -> Output: CF32[100, 256], Pad: CF32[28, 256]\n\n"

        "## Implementation\n"
        "Input -> Split at (length - size) -> Unpadded Output + Pad Output\n"
        "1. Splits the input tensor at position (length - size) along the axis.\n"
        "2. The first portion becomes the unpadded output.\n"
        "3. The second portion becomes the pad output.";
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_CORE_UNPAD_BLOCK_HH
