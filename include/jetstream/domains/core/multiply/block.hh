#ifndef JETSTREAM_DOMAINS_CORE_MULTIPLY_BLOCK_HH
#define JETSTREAM_DOMAINS_CORE_MULTIPLY_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct Multiply : public Block::Config {
    JST_BLOCK_TYPE(multiply);
    JST_BLOCK_PARAMS();
    JST_BLOCK_DESCRIPTION(
        "Multiply",
        "Element-wise multiplication.",
        "# Multiply\n"
        "The Multiply block performs element-wise multiplication of two input tensors "
        "with automatic broadcasting support. It takes two tensors of potentially "
        "different shapes and produces their product by broadcasting the smaller "
        "tensor to match the larger one's dimensions, following NumPy-style broadcasting rules.\n\n"

        "## Useful For\n"
        "- Applying gain or attenuation to signals in digital signal processing.\n"
        "- Implementing windowing functions by multiplying with window coefficients.\n"
        "- Modulating signals with element-wise multiplication.\n\n"

        "## Examples\n"
        "- Vector multiplication:\n"
        "  Input A: F32[1024] * Input B: F32[1024] -> Output: F32[1024]\n"
        "- Matrix multiplication with broadcasting:\n"
        "  Input A: CF32[512, 256] * Input B: CF32[1, 256] -> Output: CF32[512, 256]\n\n"

        "## Implementation\n"
        "Input A * Input B -> Broadcast -> Multiply -> Output\n"
        "1. Input tensors are validated for broadcasting compatibility.\n"
        "2. Output tensor shape is determined using broadcasting rules.\n"
        "3. Input tensors are broadcast to match the output shape.\n"
        "4. Element-wise multiplication is performed.";
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_CORE_MULTIPLY_BLOCK_HH
