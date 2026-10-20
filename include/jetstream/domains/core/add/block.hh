#ifndef JETSTREAM_DOMAINS_CORE_ADD_BLOCK_HH
#define JETSTREAM_DOMAINS_CORE_ADD_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct Add : public Block::Config {
    JST_BLOCK_TYPE(add);
    JST_BLOCK_DOMAIN("Core");
    JST_BLOCK_NODE_SIZE(XS);
    JST_BLOCK_PARAMS();
    JST_BLOCK_DESCRIPTION(
        "Add",
        "Adds two tensors element by element, with broadcasting.",
        "# Add\n"
        "\n"
        "Adds two tensors element by element, repeating the smaller one along missing "
        "or size-one axes. It usually mixes two signals at the same rate, or adds a "
        "fixed row or value to every batch of a stream.\n"
        "\n"
        "- **Both inputs need the same type.** Convert a real input to CF32 with a Cast"
        " block before adding a complex one.\n"
        "- **Input A sets the attributes.** Input B only fills in a `sampleRate` or "
        "`frequency` that Input A lacks or sets to 0.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Input A** (in) | F32 or CF32 | Any shape. Strided input is accepted. |\n"
        "| **Input B** (in) | Same as Input A | Aligned with Input A from the last "
        "axis, where each pair of sizes must match or one must be 1. Missing axes count"
        " as 1. Strided input is accepted. |\n"
        "| **Output** (out) | Same as the inputs | The larger size of each axis pair. "
        "Axis roles merge from both inputs, and other attributes come from Input A. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "Each row feeds Input B from a Ones Tensor with the settings shown.\n"
        "\n"
        "| Goal | Settings | Input | Output |\n"
        "|---|---|---|---|\n"
        "| Add one row to every batch | Shape [1024] | 8 batches x 1,024 samples on "
        "Input A | 8 batches x 1,024 samples |\n"
        "| Add one value everywhere | Shape [1], Data Type CF32 | 8,192 complex samples"
        " on Input A | 8,192 complex samples, each real part 1 higher |\n"
        "| Grid of sums | Shape [8, 1] | 1,024 samples on Input A | 8 x 1,024 values, "
        "with `sampleAxis` on the last axis |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The block reports that the input shapes are not broadcastable | Two aligned "
        "sizes differ and neither is 1 | Match the sizes, or give the smaller input a "
        "size of 1 on that axis. |\n"
        "| The block reports that the input data types do not match | One input is F32 "
        "and the other CF32 | Convert the real input to CF32 with a Cast block. |\n"
        "| The block reports an unsupported data type | The inputs are not F32 or CF32,"
        " such as CI8 samples from a file | Convert both with a Cast block. |\n"
        "| The block reports that signal roles map to conflicting axes | After aligning"
        " from the last axis, the inputs give one role to different axes, or two roles "
        "to one axis. A one-dimensional input counts as samples. | Match the roles with"
        " a Signal Axes block, or give a single value two axes, such as Ones Tensor "
        "Shape [1, 1]. |\n"
        "| The output reports the wrong `sampleRate` or `frequency` | Both inputs set "
        "it and Input A wins, with no warning | Connect the signal whose attributes "
        "should be kept to Input A. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "The block aligns both shapes from the last axis and reads each input through a"
        " repeating view, so nothing is copied before adding. The model below covers "
        "one buffer without attributes.\n"
        "\n"
        "```python\n"
        "import numpy as np\n"
        "\n"
        "def add(a, b):\n"
        "    if a.dtype != b.dtype:\n"
        "        raise TypeError(\"Input data types do not match.\")\n"
        "    return a + b\n"
        "```\n"
        "\n"
        "## See Also\n"
        "\n"
        "- **Multiply** combines two inputs by their product instead.\n"
        "- **Ones Tensor** makes a constant tensor for Input B.\n"
        "- **Cast** converts real input to match a complex one.\n"
        "- **Arithmetic** sums along one axis of a single tensor."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_CORE_ADD_BLOCK_HH
