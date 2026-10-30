#ifndef JETSTREAM_DOMAINS_CORE_SQUEEZE_DIMS_BLOCK_HH
#define JETSTREAM_DOMAINS_CORE_SQUEEZE_DIMS_BLOCK_HH

#include "jetstream/block.hh"
#include "jetstream/types.hh"

namespace Jetstream::Blocks {

struct SqueezeDims : public Block::Config {
    I64 axis = -1;

    JST_BLOCK_TYPE(squeeze_dims);
    JST_BLOCK_DOMAIN("Core");
    JST_BLOCK_PARAMS(axis);
    JST_BLOCK_DESCRIPTION(
        "Squeeze Dims",
        "Removes a dimension of size 1 at a specified axis.",
        "# Squeeze Dims\n"
        "\n"
        "Removes one axis of size one, without moving any data. It usually follows a "
        "block that leaves a single batch or channel, to pass the next block fewer "
        "dimensions.\n"
        "\n"
        "- **The default targets the last axis.** Most streams keep samples there, so "
        "point Axis at the size-one axis.\n"
        "- **The removed axis loses its role.** Every other role moves with its axis.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Input** (in) | Any type | Any shape with a size-one axis at Axis. The "
        "buffer must be contiguous. |\n"
        "| **Output** (out) | Same as the input | The input shape without the axis at "
        "Axis. Every other role and every attribute is kept. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **Axis** | -1 | From -N to N - 1, for N input axes, on an axis of size one |"
        " Axis to remove, counted from 0 at the outermost. Negative values count from "
        "the end. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "| Goal | Settings | Input | Output |\n"
        "|---|---|---|---|\n"
        "| Drop a single batch | Axis 0 | 1 batch x 8,192 samples | 8,192 samples |\n"
        "| Drop a trailing axis | Defaults | 1,024 samples x 1 value | 1,024 samples |\n"
        "| Drop a single channel | Axis -2 | 8 batches x 1 channel x 1,024 samples | 8 "
        "batches x 1,024 samples, with no `channelAxis` |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The block reports that it cannot squeeze a dimension whose size is not 1 | "
        "The axis at Axis is longer than one, such as the default -1 on a stream of "
        "samples | Point Axis at a size-one axis. |\n"
        "| The block reports that the axis is out of range | Axis is outside -N to N - "
        "1 for N input axes | Count axes from 0, or from -1 at the end. |\n"
        "| The block reports that it expects a contiguous tensor | The input is "
        "strided, such as a Slice with Contiguous off | Turn on Contiguous in the "
        "Slice. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "The block only removes a size-one entry from the shape and strides of the "
        "input, so no values move and nothing is computed per buffer. The model below "
        "covers one buffer.\n"
        "\n"
        "```python\n"
        "import numpy as np\n"
        "\n"
        "def squeeze_dims(x, axis=-1):\n"
        "    if not x.flags.c_contiguous:\n"
        "        raise ValueError(\"Contiguous tensor expected.\")\n"
        "    return np.squeeze(x, axis=axis)\n"
        "```\n"
        "\n"
        "## See Also\n"
        "\n"
        "- **Expand Dims** adds a size-one axis.\n"
        "- **Slice** with a single index removes an axis of any size.\n"
        "- **Reshape** sets every dimension at once."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_CORE_SQUEEZE_DIMS_BLOCK_HH
