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
        "\n"
        "Outputs a tensor of the chosen shape and type with every value set to one. It "
        "usually feeds a test chain or a block that needs a constant second input, such"
        " as Multiply.\n"
        "\n"
        "- **The values never change.** Every cycle passes the same tensor of ones "
        "downstream.\n"
        "- **The output carries no metadata.** Assign axis roles with a Signal Axes "
        "block when Shape has more than one dimension.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Output** (out) | F32, CF32, F64, or CF64, set by Data Type | Shape, with "
        "no axis roles or attributes. A one-dimensional output implies a `sampleAxis`. "
        "|\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **Shape** | [1] | One or more sizes above 0 | Output dimensions, outermost "
        "first. |\n"
        "| **Data Type** | F32 | F32, CF32, F64, CF64 | Output type. Complex values are"
        " 1 + 0i. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "| Goal | Settings | Output |\n"
        "|---|---|---|\n"
        "| One constant value | Defaults | One real value of 1 |\n"
        "| A stream of ones | Shape [8192] | 8,192 real samples of 1 |\n"
        "| Complex batches | Shape [8, 1024], CF32 | 8 x 1,024 complex values of 1 + "
        "0i, with no axis roles |\n"
        "| Double precision | Shape [4096], F64 | 4,096 real samples of 1 in double "
        "precision |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The block reports that the shape cannot be empty | Shape is [] | Enter at "
        "least one size, such as [1024]. |\n"
        "| The block reports that a shape dimension cannot be zero | Shape holds a 0 | "
        "Use sizes of 1 or more. |\n"
        "| A block downstream reports that the tensor is missing `sampleAxis` metadata "
        "| Shape has more than one dimension, and the output has no axis roles | Assign"
        " roles with a Signal Axes block. |\n"
        "\n"
        "## See Also\n"
        "\n"
        "- **Multiply** scales a signal by this tensor.\n"
        "- **Signal Axes** assigns roles to its dimensions.\n"
        "- **Signal Generator** produces a constant level as a stream."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_CORE_ONES_TENSOR_BLOCK_HH
