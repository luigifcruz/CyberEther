#ifndef JETSTREAM_DOMAINS_CORE_PERMUTATION_BLOCK_HH
#define JETSTREAM_DOMAINS_CORE_PERMUTATION_BLOCK_HH

#include <vector>

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct Permutation : public Block::Config {
    std::vector<U64> permutation = {0};

    JST_BLOCK_TYPE(permutation);
    JST_BLOCK_DOMAIN("Core");
    JST_BLOCK_PARAMS(permutation);
    JST_BLOCK_DESCRIPTION(
        "Permutation",
        "Reorders tensor axes with a user-defined permutation.",
        "# Permutation\n"
        "The Permutation block reorders tensor axes using a user-provided index "
        "list. Each output axis selects which input axis should appear in that "
        "position. The underlying tensor data is unchanged.\n\n"

        "## Arguments\n"
        "- **Permutation**: The output axis order as zero-based input axis indices.\n\n"

        "## Useful For\n"
        "- Transposing matrix-shaped tensors for downstream processing.\n"
        "- Moving channel, batch, or time axes into the order a later block expects.\n"
        "- Reusing the same tensor data with a different logical layout.\n\n"

        "## Examples\n"
        "- Transpose a 2D tensor.\n"
        "  Config: Permutation=[1, 0]\n"
        "  Input: F32[64, 1024] -> Output: F32[1024, 64]\n"
        "- Move the first axis to the end.\n"
        "  Config: Permutation=[1, 2, 0]\n"
        "  Input: CF32[8, 64, 1024] -> Output: CF32[64, 1024, 8]\n\n"

        "## Implementation\n"
        "Input -> Permutation -> Output\n"
        "1. Validates that the configuration is a full axis permutation.\n"
        "2. Reorders tensor shape and stride metadata to build the output view.\n"
        "3. The output shares the same underlying data as the input."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_CORE_PERMUTATION_BLOCK_HH
