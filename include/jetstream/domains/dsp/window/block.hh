#ifndef JETSTREAM_DOMAINS_DSP_WINDOW_BLOCK_HH
#define JETSTREAM_DOMAINS_DSP_WINDOW_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct Window : public Block::Config {
    U64 size = 1024;

    JST_BLOCK_TYPE(window);
    JST_BLOCK_DOMAIN("DSP");
    JST_BLOCK_PARAMS(size);
    JST_BLOCK_DESCRIPTION(
        "Window",
        "Generates a Blackman window function.",
        "# Window\n"
        "\n"
        "Generates a Blackman window, a taper that reduces spectral leakage in an FFT. "
        "It usually feeds a Multiply that applies it to each buffer right before the "
        "FFT.\n"
        "\n"
        "- **Size is not read from the signal.** Set it to the sample axis length of "
        "the buffer it multiplies.\n"
        "- **The window is complex.** Multiply needs a complex signal too, so convert "
        "real input with a Cast block.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Window** (out) | CF32, with zero imaginary parts | Size samples on a "
        "`sampleAxis`. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **Size** | 1,024 | 1 or more | Window length in samples. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "| Goal | Settings | Output |\n"
        "|---|---|---|\n"
        "| Taper a 1,024-sample FFT | Defaults | 1,024 values from 0 up to 1 and back, "
        "averaging 0.42 |\n"
        "| Taper 8 batches x 4,096 samples | Size 4096 | 4,096 values that Multiply "
        "applies to each batch |\n"
        "| Turn the taper off | Size 1 | One value of 1, which leaves the signal "
        "unchanged |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The block reports that the window size cannot be zero | Size is 0 | Set Size"
        " to 1 or more. |\n"
        "| Multiply reports that the input shapes are not broadcastable | Size differs "
        "from the sample axis length of the signal | Set Size to that length. |\n"
        "| Multiply reports that the input data types do not match | The signal is F32 "
        "and the window is CF32 | Convert the signal with a Cast block. |\n"
        "| Tones in the spectrum read 7.5 dB low | The window averages 0.42, which "
        "scales every tone | Multiply by 2.38 with a Multiply Constant block when "
        "absolute levels matter. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "The block evaluates the symmetric Blackman formula once, and the values never "
        "change while the flowgraph runs. The model below covers the whole output.\n"
        "\n"
        "```python\n"
        "import numpy as np\n"
        "\n"
        "def window(size=1024):\n"
        "    if size == 1:\n"
        "        return np.ones(1, dtype=np.complex64)\n"
        "    phase = 2 * np.pi * np.arange(size) / (size - 1)\n"
        "    w = 0.42 - 0.5 * np.cos(phase) + 0.08 * np.cos(2 * phase)\n"
        "    return w.astype(np.complex64)\n"
        "```\n"
        "\n"
        "## See Also\n"
        "\n"
        "- **Multiply** applies the window to the signal.\n"
        "- **Invert** turns it into a window that also centers the FFT.\n"
        "- **Spectrum Engine** windows and transforms in one block."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_DSP_WINDOW_BLOCK_HH
