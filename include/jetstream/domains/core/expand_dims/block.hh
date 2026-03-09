#ifndef JETSTREAM_DOMAINS_CORE_EXPAND_DIMS_BLOCK_HH
#define JETSTREAM_DOMAINS_CORE_EXPAND_DIMS_BLOCK_HH

#include "jetstream/block.hh"
#include "jetstream/types.hh"

namespace Jetstream::Blocks {

struct ExpandDims : public Block::Config {
    U64 axis = 0;

    JST_BLOCK_TYPE(expand_dims);
    JST_BLOCK_PARAMS(axis);
    JST_BLOCK_DESCRIPTION(
        "Expand Dims",
        "Inserts a new dimension of size 1 at a specified axis.",
        "# Expand Dims\n"
        "The Expand Dims block adds a new axis with size 1 to the tensor at the "
        "specified position. This operation creates a view into the original data "
        "without copying.\n\n"

        "## Arguments\n"
        "- **Axis**: The position where the new dimension will be inserted "
        "(0-indexed). A value of 0 inserts at the beginning.\n\n"

        "## Useful For\n"
        "- Preparing tensors for broadcasting with higher-dimensional data.\n"
        "- Adding a batch dimension to unbatched data.\n"
        "- Shaping tensors for multi-head filter operations.\n\n"

        "## Examples\n"
        "- Expand 1D to 2D at axis 0:\n"
        "  Config: Axis=0\n"
        "  Input: F32[100] -> Output: F32[1, 100]\n"
        "- Expand 1D to 2D at axis 1:\n"
        "  Config: Axis=1\n"
        "  Input: F32[100] -> Output: F32[100, 1]\n\n"

        "## Implementation\n"
        "Input -> Expand Dims -> Output\n"
        "1. Inserts a new dimension of size 1 at the specified axis.\n"
        "2. The output is a view that shares memory with the input.";
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_CORE_EXPAND_DIMS_BLOCK_HH
