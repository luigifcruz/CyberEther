#ifndef JETSTREAM_DOMAINS_CORE_EXPAND_DIMS_BLOCK_HH
#define JETSTREAM_DOMAINS_CORE_EXPAND_DIMS_BLOCK_HH

#include "jetstream/block.hh"
#include "jetstream/types.hh"

namespace Jetstream::Blocks {

struct ExpandDims : public Block::Config {
    I64 axis = -1;

    JST_BLOCK_TYPE(expand_dims);
    JST_BLOCK_DOMAIN("Core");
    JST_BLOCK_PARAMS(axis);
    JST_BLOCK_DESCRIPTION(
        "Expand Dims",
        "Inserts a new dimension of size 1 at a specified axis.",
        "# Expand Dims\n"
        "\n"
        "Inserts a new axis of size one at a chosen position, without moving any data. "
        "It usually gives a tensor the extra axis a downstream block expects, followed "
        "by Signal Axes when that axis needs a role.\n"
        "\n"
        "- **The new axis has no role.** Add a Signal Axes block when it must count as "
        "batches or channels.\n"
        "- **Axis roles move with their axes.** A `sampleAxis` pushed back by the new "
        "axis keeps its role.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Input** (in) | Any type | Any shape. The buffer must be contiguous. A "
        "one-dimensional input with no roles counts as samples. |\n"
        "| **Output** (out) | Same as the input | The input shape with a size-one axis "
        "at Axis. Every role and attribute is kept. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **Axis** | -1 | From -(N + 1) to N, for N input axes | Position of the new "
        "axis, counted from 0 at the outermost. Negative values count from the end, so "
        "-1 appends it. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "| Goal | Settings | Input | Output |\n"
        "|---|---|---|---|\n"
        "| Add a leading axis | Axis 0 | 8,192 samples | 1 x 8,192 samples |\n"
        "| Add a trailing axis | Defaults | 8,192 samples | 8,192 samples x 1 value |\n"
        "| Insert between batches and samples | Axis 1 | 8 batches x 1,024 samples | 8 "
        "batches x 1 x 1,024 samples |\n"
        "| Count from the end | Axis -2 | 8 batches x 1,024 samples | 8 batches x 1 x "
        "1,024 samples |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The block reports that the axis is out of range | Axis is outside -(N + 1) "
        "to N for N input axes | Use 0 to prepend the axis, or -1 to append it. |\n"
        "| The block reports that it expects a contiguous tensor | The input is "
        "strided, such as a Slice with Contiguous off | Turn on Contiguous in the "
        "Slice. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "The block only adds a size-one entry to the shape and strides of the input, so"
        " no values move and nothing is computed per buffer. The model below covers one"
        " buffer.\n"
        "\n"
        "```python\n"
        "import numpy as np\n"
        "\n"
        "def expand_dims(x, axis=-1):\n"
        "    if not x.flags.c_contiguous:\n"
        "        raise ValueError(\"Contiguous tensor expected.\")\n"
        "    return np.expand_dims(x, axis)\n"
        "```\n"
        "\n"
        "## See Also\n"
        "\n"
        "- **Squeeze Dims** removes a size-one axis.\n"
        "- **Signal Axes** assigns a role to the new axis.\n"
        "- **Reshape** sets every dimension at once."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_CORE_EXPAND_DIMS_BLOCK_HH
