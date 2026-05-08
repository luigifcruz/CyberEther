#ifndef JETSTREAM_DOMAINS_CORE_ADD_BLOCK_HH
#define JETSTREAM_DOMAINS_CORE_ADD_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct Add : public Block::Config {
    JST_BLOCK_TYPE(add);
    JST_BLOCK_DOMAIN("Core");
    JST_BLOCK_PARAMS();
    JST_BLOCK_DESCRIPTION(
        "Add",
        "Element-wise addition.",
        "# Add\n"
        "The Add block performs element-wise addition of two input tensors with "
        "automatic broadcasting support. It takes two tensors of potentially different "
        "shapes and produces their sum by broadcasting the smaller tensor to match "
        "the larger one's dimensions, following NumPy-style broadcasting rules.\n\n"

        "## Useful For\n"
        "- Combining signal components in digital signal processing applications.\n"
        "- Implementing offset corrections in sensor data processing.\n"
        "- Merging multiple data streams with element-wise summation.\n\n"

        "## Examples\n"
        "- Vector addition:\n"
        "  Input A: F32[1024] + Input B: F32[1024] -> Output: F32[1024]\n"
        "- Matrix addition with broadcasting:\n"
        "  Input A: CF32[512, 256] + Input B: CF32[1, 256] -> Output: CF32[512, 256]\n\n"

        "## Implementation\n"
        "Input A + Input B -> Broadcast -> Add -> Output\n"
        "1. Input tensors are validated for broadcasting compatibility.\n"
        "2. Output tensor shape is determined using broadcasting rules.\n"
        "3. Input tensors are broadcast to match the output shape.\n"
        "4. Element-wise addition is performed.";
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_CORE_ADD_BLOCK_HH
