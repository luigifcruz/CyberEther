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
        "\n"
        "Gives a tensor new dimensions while keeping its values in the same order, such"
        " as splitting a stream into rows. It usually sits between a source and blocks "
        "that expect batches, or flattens batches back into one stream.\n"
        "\n"
        "- **The default Shape is rejected.** Enter a shape before the block can run.\n"
        "- **A new shape drops the axis roles.** Add a Signal Axes block after it to "
        "assign them again.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Input** (in) | Any type | Any shape. The buffer must be contiguous unless "
        "Contiguous is on. |\n"
        "| **Output** (out) | Same as the input | Shape, filled in row-major order. "
        "Clears `sampleAxis`, `batchAxis`, and `channelAxis` unless Shape equals the "
        "input shape, and keeps other attributes. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **Shape** | [] | Bracketed sizes above 0 that multiply to the input size | "
        "Target dimensions, outermost first. |\n"
        "| **Contiguous** | Off | On, Off | Copies the input into a contiguous buffer "
        "every cycle before reshaping. Needed for strided input. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "| Goal | Settings | Input | Output |\n"
        "|---|---|---|---|\n"
        "| Split a stream into rows | Shape [8, 1024] | 8,192 samples | 8 x 1,024 "
        "values with no axis roles |\n"
        "| Flatten batches | Shape [8192] | 8 batches x 1,024 samples | 8,192 samples, "
        "with an implied `sampleAxis` |\n"
        "| Reshape a strided slice | Shape [4, 256], Contiguous on | 1,024 samples from"
        " a Slice with Contiguous off | 4 x 256 values in a new buffer |\n"
        "| Same shape, same roles | Shape [8, 1024] | 8 batches x 1,024 samples | 8 "
        "batches x 1,024 samples, roles kept |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The block reports that the shape must have at least one dimension | Shape is"
        " [], the default | Enter the target shape, such as [8, 1024]. |\n"
        "| The block reports that it cannot reshape to a different element count | The "
        "product of Shape differs from the input size | Change Shape so its product "
        "matches the input. |\n"
        "| The block reports that it cannot reshape a non-contiguous tensor | The input"
        " is strided, such as a Slice with Contiguous off | Turn on Contiguous. |\n"
        "| A block downstream reports missing signal axes | The new shape has no axis "
        "roles and more than one dimension | Assign roles with a Signal Axes block. |\n"
        "| The block reports an invalid shape syntax | Shape is missing a bracket, has "
        "a trailing comma, or holds something other than digits | Write digits "
        "separated by commas inside brackets. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "The block relabels the same memory with new dimensions in row-major order, so "
        "no values move and nothing is computed per buffer. With Contiguous on, the "
        "input is first copied into a fresh contiguous buffer on the same device. The "
        "model below covers one buffer.\n"
        "\n"
        "```python\n"
        "import numpy as np\n"
        "\n"
        "def reshape(x, shape, contiguous=False):\n"
        "    if not contiguous and not x.flags.c_contiguous:\n"
        "        raise ValueError(\"Cannot reshape non-contiguous tensor.\")\n"
        "    return np.ascontiguousarray(x).reshape(shape)\n"
        "```\n"
        "\n"
        "## See Also\n"
        "\n"
        "- **Signal Axes** assigns roles to the new dimensions.\n"
        "- **Flatten** turns any tensor into one dimension.\n"
        "- **Slice** with Contiguous on feeds it a contiguous buffer.\n"
        "- **Expand Dims** and **Squeeze Dims** add or remove a size-one axis."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_CORE_RESHAPE_BLOCK_HH
