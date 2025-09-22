#ifndef JETSTREAM_DOMAINS_DSP_OVERLAP_ADD_BLOCK_HH
#define JETSTREAM_DOMAINS_DSP_OVERLAP_ADD_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct OverlapAdd : public Block::Config {
    U64 axis = 0;

    JST_BLOCK_TYPE(overlap_add);
    JST_BLOCK_PARAMS(axis);
    JST_BLOCK_DESCRIPTION(
        "Overlap Add",
        "Sums overlap with buffer for streaming convolution.",
        "# Overlap-Add\n"
        "The Overlap Add block performs the overlap-add step of "
        "frequency-domain convolution. It takes a main buffer and "
        "an overlap region, adding the overlap from the previous "
        "batch to the current output. This enables seamless "
        "streaming FIR filtering using the overlap-add method.\n\n"

        "## Arguments\n"
        "- **Axis**: Dimension along which the overlap is "
        "applied.\n\n"

        "## Useful For\n"
        "- Overlap-add FIR filtering pipelines.\n"
        "- Streaming frequency-domain convolution.\n"
        "- Block-based signal processing with continuity.\n\n"

        "## Examples\n"
        "- 2D batched overlap-add:\n"
        "  Config: Axis=1\n"
        "  Buffer: CF32[4, 1024], Overlap: CF32[4, 50]\n"
        "  Output: CF32[4, 1024]\n\n"

        "## Implementation\n"
        "1. Copy input buffer to output.\n"
        "2. Add stored previous overlap to batch 0.\n"
        "3. Add each batch's overlap to the next batch.\n"
        "4. Store last batch's overlap for the next invocation."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_DSP_OVERLAP_ADD_BLOCK_HH
