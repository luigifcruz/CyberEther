#ifndef JETSTREAM_DOMAINS_CORE_MULTIPLY_BLOCK_HH
#define JETSTREAM_DOMAINS_CORE_MULTIPLY_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct Multiply : public Block::Config {
    JST_BLOCK_TYPE(multiply);
    JST_BLOCK_DOMAIN("Core");
    JST_BLOCK_NODE_SIZE(XS);
    JST_BLOCK_PARAMS();
    JST_BLOCK_DESCRIPTION(
        "Multiply",
        "Multiplies two tensors element by element, with broadcasting.",
        "# Multiply\n"
        "\n"
        "Multiplies two tensors element by element, repeating the smaller one along "
        "missing or size-one axes. It usually applies a Window before an FFT, or mixes "
        "a signal with a tone from a Signal Generator.\n"
        "\n"
        "- **Both inputs need the same type.** Convert a real input to CF32 with a Cast"
        " block before multiplying a complex one.\n"
        "- **Input A sets the attributes.** Input B only fills in a `sampleRate` that "
        "Input A lacks or sets to 0.\n"
        "- **The center frequencies add up.** A missing `frequency` counts as 0, so a "
        "Signal Generator leaves it unchanged.\n"
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
        "Axis roles merge from both inputs, and other attributes come from Input A. "
        "Sets `frequency` to the sum of both. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "Each row names the block that feeds Input B.\n"
        "\n"
        "| Goal | Settings | Input | Output |\n"
        "|---|---|---|---|\n"
        "| Taper one FFT buffer | Window, Size 1024 | 1,024 complex samples on Input A "
        "| 1,024 tapered complex samples |\n"
        "| Taper every batch | Window, Size 4096 | 8 batches x 4,096 complex samples on"
        " Input A | 8 batches x 4,096 tapered complex samples |\n"
        "| Move a signal down 100 kHz | Signal Generator, CF32, Frequency -0.1 MHz | "
        "8,192 complex samples at 1 MS/s on Input A | 8,192 complex samples at 1 MS/s, "
        "every component 100 kHz lower |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The block reports that the input data types do not match | One input is F32 "
        "and the other CF32, such as a real signal with a Window | Convert the real "
        "input to CF32 with a Cast block. |\n"
        "| The block reports that the input shapes are not broadcastable | Two aligned "
        "sizes differ and neither is 1, such as a Window Size that differs from the "
        "sample axis | Make the sizes match, such as setting Window Size to the sample "
        "axis length. |\n"
        "| The block reports an unsupported data type | The inputs are not F32 or CF32,"
        " such as CI8 samples from a file | Convert both with a Cast block. |\n"
        "| The block reports that signal roles map to conflicting axes | After aligning"
        " from the last axis, the inputs give one role to different axes, or two roles "
        "to one axis. A one-dimensional input counts as samples. | Match the roles with"
        " a Signal Axes block, or give a single value two axes, such as Ones Tensor "
        "Shape [1, 1]. |\n"
        "| The output reports the wrong `sampleRate` | Both inputs set it and Input A "
        "wins, with no warning | Connect the signal whose rate should be kept to Input "
        "A. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "The block aligns both shapes from the last axis and reads each input through a"
        " repeating view, so nothing is copied before multiplying. The model below "
        "covers one buffer without attributes.\n"
        "\n"
        "```python\n"
        "import numpy as np\n"
        "\n"
        "def multiply(a, b):\n"
        "    if a.dtype != b.dtype:\n"
        "        raise TypeError(\"Input data types do not match.\")\n"
        "    return a * b\n"
        "```\n"
        "\n"
        "## See Also\n"
        "\n"
        "- **Window** makes the taper it applies before an FFT.\n"
        "- **Signal Generator** makes the tone for mixing.\n"
        "- **Multiply Constant** scales by one number without a second input.\n"
        "- **Add** combines two inputs by their sum instead."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_CORE_MULTIPLY_BLOCK_HH
