#ifndef JETSTREAM_DOMAINS_CORE_FLATTEN_BLOCK_HH
#define JETSTREAM_DOMAINS_CORE_FLATTEN_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct Flatten : public Block::Config {
    bool contiguous = false;

    JST_BLOCK_TYPE(flatten);
    JST_BLOCK_DOMAIN("Core");
    JST_BLOCK_PARAMS(contiguous);
    JST_BLOCK_DESCRIPTION(
        "Flatten",
        "Flattens a tensor to one dimension.",
        "# Flatten\n"
        "\n"
        "Turns a tensor of any shape into one dimension, keeping its values in "
        "row-major order. It usually sits before blocks that expect a single stream, "
        "such as joining batches into one run of samples.\n"
        "\n"
        "- **Axis roles are dropped.** Blocks downstream read the single dimension as "
        "samples.\n"
        "- **The last axis varies fastest.** Two channels of samples come out one "
        "channel after the other.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Input** (in) | Any type | Any shape. The buffer must be contiguous unless "
        "Contiguous is on. |\n"
        "| **Output** (out) | Same as the input | One dimension holding every value. "
        "Clears `sampleAxis`, `batchAxis`, and `channelAxis` unless the input is "
        "one-dimensional, and keeps other attributes. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **Contiguous** | Off | On, Off | Copies the input into a contiguous buffer "
        "every cycle before flattening. Needed for strided input. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "| Goal | Settings | Input | Output |\n"
        "|---|---|---|---|\n"
        "| Join batches into one stream | Defaults | 8 batches x 1,024 samples | 8,192 "
        "samples |\n"
        "| Join channels end to end | Defaults | 2 channels x 4,096 samples | 8,192 "
        "samples, channel 0 first |\n"
        "| Interleave batches | Contiguous on | 1,024 samples x 8 batches from a "
        "Permutation | 8,192 samples, one from each batch in turn |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The block reports that it expects a contiguous tensor | The input is "
        "strided, such as a Slice or Permutation with Contiguous off | Turn on "
        "Contiguous. |\n"
        "| Channels follow each other instead of interleaving | The last axis varies "
        "fastest, and it holds the samples | Put the channels last with a Permutation "
        "first, and turn on Contiguous. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "The block relabels the same memory as one dimension, so no values move and "
        "nothing is computed per buffer. With Contiguous on, the input is first copied "
        "into a fresh contiguous buffer on the same device. The model below covers one "
        "buffer.\n"
        "\n"
        "```python\n"
        "import numpy as np\n"
        "\n"
        "def flatten(x, contiguous=False):\n"
        "    if contiguous:\n"
        "        x = np.ascontiguousarray(x)\n"
        "    if not x.flags.c_contiguous:\n"
        "        raise ValueError(\"Contiguous tensor expected.\")\n"
        "    return x.reshape(-1)\n"
        "```\n"
        "\n"
        "## See Also\n"
        "\n"
        "- **Reshape** sets any shape, not only one dimension.\n"
        "- **Permutation** reorders axes before flattening.\n"
        "- **Slice** keeps one channel instead of joining them."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_CORE_FLATTEN_BLOCK_HH
