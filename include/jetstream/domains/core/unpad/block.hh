#ifndef JETSTREAM_DOMAINS_CORE_UNPAD_BLOCK_HH
#define JETSTREAM_DOMAINS_CORE_UNPAD_BLOCK_HH

#include "jetstream/block.hh"
#include "jetstream/types.hh"

namespace Jetstream::Blocks {

struct Unpad : public Block::Config {
    U64 size = 0;
    I64 axis = -1;

    JST_BLOCK_TYPE(unpad);
    JST_BLOCK_DOMAIN("Core");
    JST_BLOCK_PARAMS(size, axis);
    JST_BLOCK_DESCRIPTION(
        "Unpad",
        "Removes padding from a tensor.",
        "# Unpad\n"
        "\n"
        "Splits a fixed number of entries off the end of one axis, and outputs both the"
        " trimmed tensor and the removed tail. It usually follows the inverse FFT of a "
        "padded convolution, and feeds both parts to an Overlap Add.\n"
        "\n"
        "- **Pad Size counts entries, not the final length.** Trimming 1,024 samples to"
        " 1,000 takes Pad Size 24.\n"
        "- **The default leaves Pad empty.** A block wired to Pad then reports a "
        "zero-size input.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Input** (in) | F32 or CF32 | Any shape. The buffer must be contiguous. |\n"
        "| **Output** (out) | Same as the input | The input shape with Pad Size removed"
        " from the end of Pad Axis. Every role and attribute is kept. |\n"
        "| **Pad** (out) | Same as the input | The removed tail, Pad Size long on Pad "
        "Axis. Every role and attribute is kept. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **Pad Size** | 0 | 0 up to the length of Pad Axis | Entries moved from the "
        "end of Pad Axis to Pad. |\n"
        "| **Pad Axis** | -1 | From -N to N - 1, for N input axes | Axis that shrinks, "
        "counted from 0 at the outermost. Negative values count from the end. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "| Goal | Settings | Input | Output |\n"
        "|---|---|---|---|\n"
        "| Undo padding for a power-of-two FFT | Pad Size 24 | 1,024 samples | Output "
        "1,000 samples, Pad 24 samples |\n"
        "| Split a convolution tail | Pad Size 10 | 8 batches x 2 channels x 810 "
        "samples | Output 8 batches x 2 channels x 800 samples, Pad 8 batches x 2 "
        "channels x 10 samples |\n"
        "| Drop trailing batches | Pad Size 8, Pad Axis 0 | 16 batches x 1,024 samples "
        "| Output 8 batches x 1,024 samples, Pad 8 batches x 1,024 samples |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The block reports that the size exceeds the axis dimension | Pad Size is "
        "larger than the length of Pad Axis | Lower Pad Size, or point Pad Axis at the "
        "padded axis. |\n"
        "| A block downstream reports that its input size is zero | Pad Size is 0, "
        "which empties Pad, or equals the axis length, which empties Output | Set Pad "
        "Size above 0 and below the axis length, or leave the empty port unconnected. |\n"
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
        "Each row along Pad Axis is split where the tail begins, and both parts are "
        "copied into their own buffers. The model below covers one buffer.\n"
        "\n"
        "```python\n"
        "import numpy as np\n"
        "\n"
        "def unpad(x, size=0, axis=-1):\n"
        "    if size > x.shape[axis]:\n"
        "        raise ValueError(\"Size exceeds axis dimension.\")\n"
        "    output, pad = np.split(x, [x.shape[axis] - size], axis=axis)\n"
        "    return output, pad\n"
        "```\n"
        "\n"
        "## See Also\n"
        "\n"
        "- **Overlap Add** takes Output and Pad as its two inputs.\n"
        "- **Pad** appends the zeros this block removes.\n"
        "- **Slice** keeps a range without a second output."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_CORE_UNPAD_BLOCK_HH
