#ifndef JETSTREAM_DOMAINS_DSP_INVERT_BLOCK_HH
#define JETSTREAM_DOMAINS_DSP_INVERT_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct Invert : public Block::Config {
    JST_BLOCK_TYPE(invert);
    JST_BLOCK_DOMAIN("DSP");
    JST_BLOCK_NODE_SIZE(XS);
    JST_BLOCK_PARAMS();
    JST_BLOCK_DESCRIPTION(
        "Invert",
        "Shifts a signal so that the following FFT is centered.",
        "# Invert\n"
        "\n"
        "Shifts the spectrum by half the band, so that a following FFT puts 0 Hz in the"
        " middle instead of the first bin. It usually sits right before an FFT, or "
        "shapes a Window that then multiplies the signal.\n"
        "\n"
        "- **Each lane is independent.** Every batch and channel is processed "
        "separately.\n"
        "- **Spectrum blocks already shift.** Spectrum Analyzer and Spectrum Engine "
        "center their own FFT, so leave it out before them.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Input** (in) | F32 or CF32 | Needs a `sampleAxis`, implied for "
        "one-dimensional input. A `batchAxis` and a `channelAxis` are optional. |\n"
        "| **Output** (out) | CF32 | Same shape, with every axis and attribute kept. |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The block reports that the input must contain valid signal axis metadata | "
        "The input has more than one dimension and no `sampleAxis` | Assign roles with "
        "a Signal Axes block. |\n"
        "| The block reports an unsupported data type | The input is not F32 or CF32 | "
        "Convert it with a Cast block. |\n"
        "| The spectrum is still not centered | The block follows the FFT instead of "
        "preceding it | Move it before the FFT. |\n"
        "| A spectrum view shows 0 Hz at its edges | Spectrum Analyzer or Spectrum "
        "Engine shifts a second time | Remove the block. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "Multiplying the samples by a steady rotation moves every frequency by half the"
        " band, the same result as swapping the spectrum halves. Even lengths flip the "
        "sign of every other sample, and odd lengths shift by the whole bin just below "
        "half. The model below covers one lane.\n"
        "\n"
        "```python\n"
        "import numpy as np\n"
        "\n"
        "def invert(x):\n"
        "    n = np.arange(len(x))\n"
        "    shift = len(x) // 2\n"
        "    return (x * np.exp(2j * np.pi * shift * n / len(x))).astype(np.complex64)\n"
        "```\n"
        "\n"
        "## See Also\n"
        "\n"
        "- The centered spectrum comes from the **FFT** that follows it.\n"
        "- **Window** is often inverted once and multiplied into the signal.\n"
        "- **Spectrum Engine** windows, shifts, and transforms in one block."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_DSP_INVERT_BLOCK_HH
