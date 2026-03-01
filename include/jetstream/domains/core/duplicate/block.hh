#ifndef JETSTREAM_DOMAINS_CORE_DUPLICATE_BLOCK_HH
#define JETSTREAM_DOMAINS_CORE_DUPLICATE_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct Duplicate : public Block::Config {
    bool hostAccessible = true;

    JST_BLOCK_TYPE(duplicate);
    JST_BLOCK_PARAMS(hostAccessible);
    JST_BLOCK_DESCRIPTION(
        "Duplicate",
        "Copies and transfers signal data.",
        "# Duplicate\n"
        "The Duplicate block copies the input signal to a new output buffer. "
        "This is useful for converting non-contiguous buffers to contiguous ones, "
        "or for transferring data between host and device memory.\n\n"

        "## Arguments\n"
        "- **Host Accessible**: When enabled, the output buffer can be accessed "
        "from the CPU. This is useful when GPU data needs to be read by the host.\n\n"

        "## Useful For\n"
        "- Converting non-contiguous memory to contiguous.\n"
        "- Creating host-accessible copies of GPU data.\n"
        "- Data transfer between different memory spaces.\n\n"

        "## Examples\n"
        "- Duplicate a GPU tensor for CPU access:\n"
        "  Config: Host Accessible=true\n"
        "  Input: CF32[8192] -> Output: CF32[8192] (host-accessible copy)\n\n"

        "## Implementation\n"
        "Input Buffer -> Memory Copy -> Output Buffer\n"
        "1. Allocates output buffer with same shape.\n"
        "2. Copies input data to output.\n"
        "3. Output has same shape and type as input.";
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_CORE_DUPLICATE_BLOCK_HH
