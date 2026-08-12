#ifndef JETSTREAM_DOMAINS_CORE_FLATTEN_BLOCK_HH
#define JETSTREAM_DOMAINS_CORE_FLATTEN_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct Flatten : public Block::Config {
    bool contiguous = false;

    JST_BLOCK_TYPE(flatten);
    JST_BLOCK_DOMAIN("Core");
    JST_BLOCK_PARAMS(contiguous);
    JST_BLOCK_DESCRIPTION(
        "Flatten",
        "Flattens a tensor to one dimension.",
        "# Flatten\n"
        "The Flatten block converts any tensor shape into a 1D tensor without "
        "modifying its data. By default it creates a view into the original data "
        "without copying, so the input must be contiguous. Enable Contiguous to "
        "copy non-contiguous inputs before flattening.\n\n"

        "## Arguments\n"
        "- **Contiguous**: When enabled, copies data to ensure contiguous memory "
        "layout before flattening.\n\n"

        "## Useful For\n"
        "- Preparing multidimensional data for blocks that expect 1D input.\n"
        "- Collapsing batched or image-like tensors into a linear stream.\n"
        "- Removing shape structure while preserving element order.\n\n"

        "## Examples\n"
        "- Flatten 2D to 1D:\n"
        "  Input: CF32[100, 256] -> Output: CF32[25600]\n"
        "- Flatten 3D to 1D:\n"
        "  Input: F32[4, 8, 16] -> Output: F32[512]\n\n"

        "## Implementation\n"
        "Input -> (Duplicate) -> Flatten -> Output\n"
        "1. If Contiguous is enabled, a Duplicate module copies the data first.\n"
        "2. Flatten creates a 1D view with the same number of elements."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_CORE_FLATTEN_BLOCK_HH
