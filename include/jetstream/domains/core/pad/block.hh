#ifndef JETSTREAM_DOMAINS_CORE_PAD_BLOCK_HH
#define JETSTREAM_DOMAINS_CORE_PAD_BLOCK_HH

#include "jetstream/block.hh"
#include "jetstream/types.hh"

namespace Jetstream::Blocks {

struct Pad : public Block::Config {
    U64 size = 0;
    I64 axis = -1;

    JST_BLOCK_TYPE(pad);
    JST_BLOCK_DOMAIN("Core");
    JST_BLOCK_PARAMS(size, axis);
    JST_BLOCK_DESCRIPTION(
        "Pad",
        "Adds zeros to the end of a tensor.",
        "# Pad\n"
        "\n"
        "Appends zeros to the end of one axis, making each row along it longer. It "
        "usually sits before an FFT or a convolution that needs a longer buffer, and an"
        " Unpad later removes the extra.\n"
        "\n"
        "- **Pad Size counts zeros, not the final length.** Padding 1,000 samples to "
        "1,024 takes Pad Size 24.\n"
        "- **The default Pad Size adds nothing.** The output is then a copy of the "
        "input.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Input** (in) | F32 or CF32 | Any shape. The buffer must be contiguous. |\n"
        "| **Output** (out) | Same as the input | The input shape with Pad Size added "
        "to Pad Axis. Every role and attribute is kept. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **Pad Size** | 0 | 0 or more | Zeros added to the end of Pad Axis. |\n"
        "| **Pad Axis** | -1 | From -N to N - 1, for N input axes | Axis that grows, "
        "counted from 0 at the outermost. Negative values count from the end. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "| Goal | Settings | Input | Output |\n"
        "|---|---|---|---|\n"
        "| Power-of-two FFT length | Pad Size 24 | 1,000 samples | 1,024 samples, the "
        "last 24 zero |\n"
        "| Room for an FFT convolution | Pad Size 100 | 8 batches x 8,000 samples | 8 "
        "batches x 8,100 samples |\n"
        "| Match the taps to that length | Pad Size 7999 | 2 heads x 101 taps | 2 heads"
        " x 8,100 taps |\n"
        "| Add empty batches | Pad Size 8, Pad Axis 0 | 8 batches x 1,024 samples | 16 "
        "batches x 1,024 samples, the last 8 zero |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The block reports an unsupported data type | The input is not F32 or CF32 | "
        "Convert it with a Cast block. |\n"
        "| The block reports that the axis is out of range | Pad Axis is outside -N to "
        "N - 1 for N input axes | Count axes from 0, or from -1 at the end. |\n"
        "| The block reports that it expects a contiguous tensor | The input is "
        "strided, such as a Slice with Contiguous off | Turn on Contiguous in the "
        "Slice. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "Each row along Pad Axis is copied to the start of a longer row, and the rest "
        "of that row is filled with zeros. The model below covers one buffer.\n"
        "\n"
        "```python\n"
        "import numpy as np\n"
        "\n"
        "def pad(x, size=0, axis=-1):\n"
        "    widths = [(0, 0)] * x.ndim\n"
        "    widths[axis] = (0, size)\n"
        "    return np.pad(x, widths)\n"
        "```\n"
        "\n"
        "## See Also\n"
        "\n"
        "- **Unpad** removes the padding and returns the tail.\n"
        "- The padded buffer often feeds an **FFT**.\n"
        "- **Overlap Add** joins the convolved batches after an Unpad."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_CORE_PAD_BLOCK_HH
