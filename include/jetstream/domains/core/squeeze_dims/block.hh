#ifndef JETSTREAM_DOMAINS_CORE_SQUEEZE_DIMS_BLOCK_HH
#define JETSTREAM_DOMAINS_CORE_SQUEEZE_DIMS_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct SqueezeDims : public Block::Config {
    U64 axis = 0;

    JST_BLOCK_TYPE(squeeze_dims);
    JST_BLOCK_PARAMS(axis);
    JST_BLOCK_DESCRIPTION(
        "Squeeze Dims",
        "Removes a dimension of size 1 at a specified axis.",
        "# Squeeze Dims\n"
        "The Squeeze Dims block removes an axis with size 1 from the tensor at "
        "the specified position. This operation creates a view into the original "
        "data without copying. The dimension at the specified position must have "
        "size 1.\n\n"

        "## Arguments\n"
        "- **Axis**: The position of the dimension to remove (0-indexed). The "
        "dimension at this position must have size 1.\n\n"

        "## Useful For\n"
        "- Removing singleton dimensions after reduction operations.\n"
        "- Converting batched single-element tensors to unbatched form.\n"
        "- Simplifying tensor shapes for downstream processing.\n\n"

        "## Examples\n"
        "- Squeeze 2D to 1D at axis 0:\n"
        "  Config: Axis=0\n"
        "  Input: F32[1, 100] -> Output: F32[100]\n"
        "- Squeeze 3D to 2D at axis 1:\n"
        "  Config: Axis=1\n"
        "  Input: F32[10, 1, 20] -> Output: F32[10, 20]\n\n"

        "## Implementation\n"
        "Input -> Squeeze Dims -> Output\n"
        "1. Removes the dimension of size 1 at the specified axis.\n"
        "2. The output is a view that shares memory with the input.";
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_CORE_SQUEEZE_DIMS_BLOCK_HH
