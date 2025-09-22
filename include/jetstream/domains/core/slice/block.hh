#ifndef JETSTREAM_DOMAINS_CORE_SLICE_BLOCK_HH
#define JETSTREAM_DOMAINS_CORE_SLICE_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct Slice : public Block::Config {
    std::string slice = "[...]";
    bool contiguous = false;

    JST_BLOCK_TYPE(slice);
    JST_BLOCK_PARAMS(slice, contiguous);
    JST_BLOCK_DESCRIPTION(
        "Slice",
        "Extracts a subset of a tensor.",
        "# Slice\n"
        "The Slice block creates a view into a tensor using NumPy-like slicing "
        "syntax. Supports `[n]` for indexing, `[start:stop]` for ranges, "
        "`[start:stop:step]` for strided ranges, `[:]` for all elements, and "
        "`[...]` for remaining dimensions.\n\n"

        "## Arguments\n"
        "- **Slice**: The slice notation string (e.g., `[0:10, ...]`).\n"
        "- **Contiguous**: When enabled, copies data to ensure contiguous memory layout.\n\n"

        "## Useful For\n"
        "- Extracting subsets of data without copying.\n"
        "- Selecting specific channels or time ranges from batched tensors.\n"
        "- Downsampling via strided slicing.\n\n"

        "## Examples\n"
        "- Extract first 100 samples:\n"
        "  Config: Slice=[0:100]\n"
        "  Input: F32[1024] -> Output: F32[100]\n"
        "- Extract first channel from batched data:\n"
        "  Config: Slice=[0, ...]\n"
        "  Input: CF32[8, 1024] -> Output: CF32[1024]\n\n"

        "## Implementation\n"
        "Input -> Slice -> (Duplicate) -> Output\n"
        "1. Slice module creates a view using the specified notation.\n"
        "2. If Contiguous is enabled, a Duplicate module copies the sliced data.\n"
        "3. By default the output shares memory with the input.";
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_CORE_SLICE_BLOCK_HH
