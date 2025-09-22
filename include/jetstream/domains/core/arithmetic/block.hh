#ifndef JETSTREAM_DOMAINS_CORE_ARITHMETIC_BLOCK_HH
#define JETSTREAM_DOMAINS_CORE_ARITHMETIC_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct Arithmetic : public Block::Config {
    std::string operation = "add";
    U64 axis = 0;
    bool squeeze = false;

    JST_BLOCK_TYPE(arithmetic);
    JST_BLOCK_PARAMS(operation, axis, squeeze);
    JST_BLOCK_DESCRIPTION(
        "Arithmetic",
        "Reduces a tensor along an axis using an arithmetic operation.",
        "# Arithmetic\n"
        "The Arithmetic block performs a reduction operation along a specified axis of "
        "the input tensor. It supports addition, subtraction, multiplication, and "
        "division as reduction operations. The specified axis is collapsed to size 1 "
        "in the output, and optionally squeezed to remove the dimension entirely.\n\n"

        "## Arguments\n"
        "- **Operation**: The arithmetic operation to apply (add, sub, mul, div).\n"
        "- **Axis**: The axis along which to reduce.\n"
        "- **Squeeze**: Whether to remove the reduced dimension from the output.\n\n"

        "## Useful For\n"
        "- Summing signal components along a specific dimension.\n"
        "- Computing averages when combined with division.\n"
        "- Reducing multi-dimensional data to lower dimensions.\n\n"

        "## Examples\n"
        "- Sum reduction:\n"
        "  Input: F32[4, 8], axis=1, operation=add -> Output: F32[4, 1]\n"
        "- Sum reduction with squeeze:\n"
        "  Input: F32[4, 8], axis=1, operation=add, squeeze=true -> Output: F32[4]\n\n"

        "## Implementation\n"
        "Input -> Reduce(axis, operation) -> Output\n"
        "1. Output tensor is allocated with the reduced axis set to size 1.\n"
        "2. Output is broadcast back to input shape for element-wise accumulation.\n"
        "3. The reduction operation is applied across the specified axis.\n"
        "4. Optionally, the reduced dimension is squeezed from the output.";
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_CORE_ARITHMETIC_BLOCK_HH
