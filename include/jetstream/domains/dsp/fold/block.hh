#ifndef JETSTREAM_DOMAINS_DSP_FOLD_BLOCK_HH
#define JETSTREAM_DOMAINS_DSP_FOLD_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct Fold : public Block::Config {
    U64 axis = 0;
    U64 offset = 0;
    U64 size = 0;

    JST_BLOCK_TYPE(fold);
    JST_BLOCK_DOMAIN("DSP");
    JST_BLOCK_PARAMS(axis, offset, size);
    JST_BLOCK_DESCRIPTION(
        "Fold",
        "Folds the input signal along a specified axis.",
        "# Fold\n"
        "The Fold block accumulates (folds) signal data along a given axis, "
        "reducing its size by a decimation factor. Each output element is the "
        "average of the corresponding folded input elements. An optional offset "
        "shifts the folding origin along the axis.\n\n"

        "## Arguments\n"
        "- **Axis**: Dimension along which to fold.\n"
        "- **Offset**: Sample offset applied before folding.\n"
        "- **Size**: Output size along the folded axis. Must evenly "
        "divide the input size along that axis.\n\n"

        "## Useful For\n"
        "- Overlap-and-add signal processing.\n"
        "- Period averaging of cyclic signals.\n"
        "- Decimation with averaging.\n\n"

        "## Examples\n"
        "- Fold a 1D signal of 8192 samples into 1024:\n"
        "  Config: Axis=0, Offset=0, Size=1024\n"
        "  Input: CF32[8192] -> Output: CF32[1024]\n\n"

        "## Implementation\n"
        "1. Zero the output buffer.\n"
        "2. For each input element, apply offset and modulo fold.\n"
        "3. Accumulate into the output.\n"
        "4. Divide by the decimation factor to compute the average."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_DSP_FOLD_BLOCK_HH
