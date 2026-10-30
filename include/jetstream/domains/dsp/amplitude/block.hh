#ifndef JETSTREAM_DOMAINS_DSP_AMPLITUDE_BLOCK_HH
#define JETSTREAM_DOMAINS_DSP_AMPLITUDE_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct Amplitude : public Block::Config {
    JST_BLOCK_TYPE(amplitude);
    JST_BLOCK_DOMAIN("DSP");
    JST_BLOCK_NODE_SIZE(XS);
    JST_BLOCK_PARAMS();
    JST_BLOCK_DESCRIPTION(
        "Amplitude",
        "Calculates the amplitude of a signal in decibels.",
        "# Amplitude\n"
        "\n"
        "Converts each value to its magnitude in decibels, scaled so a spectrum reads "
        "in dBFS. It usually follows an FFT, and feeds a Range block or a plot.\n"
        "\n"
        "- **Readings are in dBFS after an FFT.** A full-scale tone reads 0 dB in its "
        "bin, whatever the FFT length.\n"
        "- **Channel-only input is not scaled.** Without a `sampleAxis`, values become "
        "plain decibels.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Input** (in) | F32 or CF32 | Needs a `sampleAxis` or a `channelAxis`, with"
        " a `sampleAxis` implied for one-dimensional input. Strided input is accepted. "
        "|\n"
        "| **Output** (out) | F32 | Same shape, with every axis and attribute kept. "
        "Exact zeros become negative infinity. |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The block reports an unsupported input data type | The input is not F32 or "
        "CF32, such as CI16 samples from a file | Convert it with a Cast block. |\n"
        "| The block reports that the input needs a `sampleAxis` or a `channelAxis` | A"
        " multi-dimensional input has no axis metadata | Mark the axes with a Signal "
        "Axes block. |\n"
        "| The block reports that the input needs valid signal axis metadata | Two "
        "roles share an axis, or one points past the last axis | Fix the roles with a "
        "Signal Axes block. |\n"
        "| Time samples read far below their level | The scaling assumes FFT bins, so "
        "8,192 samples lose 78.3 dB | Multiply the input by the sample axis length with"
        " Multiply Constant. |\n"
        "| A Waterfall shows one flat color | Decibel values sit below the 0 to 1 color"
        " range | Map them with a Range block. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "Each magnitude goes through a fast polynomial approximation of the logarithm. "
        "A fixed offset tied to the sample axis length then makes a full-scale tone in "
        "an unnormalized FFT read as full scale. The model below covers one buffer "
        "along the sample axis.\n"
        "\n"
        "```python\n"
        "import numpy as np\n"
        "\n"
        "def amplitude(x):\n"
        "    with np.errstate(divide=\"ignore\"):\n"
        "        return 20 * np.log10(np.abs(x)) - 20 * np.log10(len(x))\n"
        "```\n"
        "\n"
        "## See Also\n"
        "\n"
        "- Spectra to convert usually come from an **FFT**.\n"
        "- **Spectrum Engine** windows, transforms, and scales in one block.\n"
        "- **Range** maps decibels onto 0 to 1 for a Waterfall.\n"
        "- **Signal Axes** marks the axes it needs."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_DSP_AMPLITUDE_BLOCK_HH
