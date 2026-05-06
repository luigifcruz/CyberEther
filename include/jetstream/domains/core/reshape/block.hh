#ifndef JETSTREAM_DOMAINS_CORE_RESHAPE_BLOCK_HH
#define JETSTREAM_DOMAINS_CORE_RESHAPE_BLOCK_HH

#include <string>

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct Reshape : public Block::Config {
    std::string shape = "[]";
    bool contiguous = false;

    JST_BLOCK_TYPE(reshape);
    JST_BLOCK_DOMAIN("Core");
    JST_BLOCK_PARAMS(shape, contiguous);
    JST_BLOCK_DESCRIPTION(
        "Reshape",
        "Changes the shape of a tensor.",
        "# Reshape\n"
        "The Reshape block changes the dimensions of a tensor without modifying "
        "its data. The total number of elements must remain the same. By default "
        "it creates a view into the original data without copying. The shape string "
        "uses bracket notation (e.g., `[10, 10]`, `[2, 5, 10]`).\n\n"

        "## Arguments\n"
        "- **Shape**: The target shape as a bracketed comma-separated list "
        "(e.g., `[100, 200]`).\n"
        "- **Contiguous**: When enabled, copies data to ensure contiguous memory "
        "layout before reshaping.\n\n"

        "## Useful For\n"
        "- Flattening multi-dimensional data for downstream processing.\n"
        "- Splitting a 1D stream into batched 2D tensors.\n"
        "- Rearranging tensor dimensions for broadcasting compatibility.\n\n"

        "## Examples\n"
        "- Flatten 2D to 1D:\n"
        "  Config: Shape=[25600]\n"
        "  Input: CF32[100, 256] -> Output: CF32[25600]\n"
        "- Unflatten 1D to 2D:\n"
        "  Config: Shape=[32, 32]\n"
        "  Input: F32[1024] -> Output: F32[32, 32]\n\n"

        "## Implementation\n"
        "Input -> (Duplicate) -> Reshape -> Output\n"
        "1. If Contiguous is enabled, a Duplicate module copies the data first.\n"
        "2. Reshape module creates a view with the new shape.\n"
        "3. The product of all new dimensions must equal the total element count.";
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_CORE_RESHAPE_BLOCK_HH
