#ifndef JETSTREAM_DOMAINS_CORE_ONES_TENSOR_BLOCK_HH
#define JETSTREAM_DOMAINS_CORE_ONES_TENSOR_BLOCK_HH

#include <string>
#include <vector>

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct OnesTensor : public Block::Config {
    std::vector<U64> shape = {1};
    std::string dataType = "F32";

    JST_BLOCK_TYPE(ones_tensor);
    JST_BLOCK_DOMAIN("Core");
    JST_BLOCK_PARAMS(shape, dataType);
    JST_BLOCK_DESCRIPTION(
        "Ones Tensor",
        "Creates a tensor filled with ones.",
        "# Ones Tensor\n"
        "The Ones Tensor block creates a tensor with a user-defined shape and data type, fills every element with the multiplicative identity, and serves that tensor as a source output. "
        "For F32 this is `1.0`, and for CF32 this is `1+0i`.\n\n"

        "## Arguments\n"
        "- **Shape**: The output tensor dimensions as a list of positive integers.\n"
        "- **Data Type**: The output tensor format. Supported values are F32 and CF32.\n\n"

        "## Useful For\n"
        "- Creating reference tensors for tests and demos.\n"
        "- Feeding multiplicative identity tensors into graph branches.\n"
        "- Initializing downstream pipelines with a known tensor of ones.\n\n"

        "## Examples\n"
        "- Create a real 1D tensor.\n"
        "  Config: Shape=[4096], DataType=F32.\n"
        "  Output: F32[4096].\n"
        "- Create a complex 2D tensor.\n"
        "  Config: Shape=[64, 128], DataType=CF32.\n"
        "  Output: CF32[64, 128].\n\n"

        "## Implementation\n"
        "Ones -> Tensor Allocation -> Output.\n"
        "1. Allocates the output tensor on the selected device.\n"
        "2. Fills every element with `1.0` for F32 or `1+0i` for CF32.\n"
        "3. Exposes the tensor as the block output."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_CORE_ONES_TENSOR_BLOCK_HH
