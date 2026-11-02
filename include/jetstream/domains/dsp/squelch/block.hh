#ifndef JETSTREAM_DOMAINS_DSP_SQUELCH_BLOCK_HH
#define JETSTREAM_DOMAINS_DSP_SQUELCH_BLOCK_HH

#include "jetstream/block.hh"

// TODO: Add smoothing to the threshold/amplitude state.
// TODO: Add hysteresis to reduce chatter around the threshold.
// TODO: Add a "Set Threshold" button in the UI.

namespace Jetstream::Blocks {

struct Squelch : public Block::Config {
    F32 threshold = 0.1f;

    JST_BLOCK_TYPE(squelch);
    JST_BLOCK_DOMAIN("DSP");
    JST_BLOCK_PARAMS(threshold);
    JST_BLOCK_DESCRIPTION(
        "Squelch",
        "Passes input only when signal strength is above a threshold.",
        "# Squelch\n"
        "\n"
        "Passes each buffer through unchanged when its strongest sample is above "
        "Threshold, and drops it otherwise. It usually sits between a Filter that "
        "isolates one channel and a demodulator, to mute the noise between "
        "transmissions.\n"
        "\n"
        "- **Closed buffers stop the chain.** Blocks downstream do not run until the "
        "squelch opens again.\n"
        "- **One decision covers every head.** The loudest sample across all heads and "
        "batches opens or closes them together.\n"
        "- **Threshold is a linear amplitude.** Compare it with the Signal Level "
        "readout, not with a dB value.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Input** (in) | F32 or CF32 | Any shape. The buffer must be contiguous. |\n"
        "| **Output** (out) | Same as the input | The input buffer itself, with every "
        "axis and attribute kept. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **Threshold** | 0.1 | 0 or more | The squelch opens when Signal Level is "
        "above this value. |\n"
        "\n"
        "## Readouts\n"
        "\n"
        "| Readout | Shows |\n"
        "|---|---|\n"
        "| **Squelch** | Open when the latest buffer passed, Closed otherwise. |\n"
        "| **Signal Level** | Peak magnitude of the latest buffer, on the Threshold "
        "scale. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "| Goal | Settings | Input | Output |\n"
        "|---|---|---|---|\n"
        "| Station on air | Threshold 0.3 | 819 samples with a peak of 0.8 | The same "
        "819 samples |\n"
        "| Station off air | Threshold 0.3 | 819 samples with a peak of 0.05 | Nothing,"
        " the chain skips the buffer |\n"
        "| Drop only silence | Threshold 0 | 8,192 samples with a peak of 0.001 | The "
        "same 8,192 samples |\n"
        "| Three stations, one gate | Threshold 0.3 | 3 heads x 819 samples, one head "
        "peaking at 0.8 | All 3 heads x 819 samples |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| Blocks downstream stop updating and Squelch shows Closed | Signal Level is "
        "at or below Threshold | Lower Threshold below the Signal Level of the wanted "
        "signal. |\n"
        "| Noise still comes through | Noise peaks rise above Threshold | Raise "
        "Threshold above the Signal Level shown with no signal. |\n"
        "| Audio cuts in and out on a weak signal | Each buffer is judged alone, with "
        "no hysteresis | Set Threshold midway between the noise and signal levels. |\n"
        "| The block reports an unsupported data type | The input is not F32 or CF32 | "
        "Convert it with a Cast block. |\n"
        "| The block reports that it expects a contiguous tensor | The input is "
        "strided, such as a Slice with Contiguous off | Turn on Contiguous in the "
        "Slice. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "The block finds the largest sample magnitude in each buffer and compares it "
        "with the threshold, with no smoothing or hysteresis between buffers. An open "
        "buffer is handed on without a copy, and a closed one tells the scheduler to "
        "skip every block downstream for that cycle. The model below covers one buffer "
        "and returns nothing when closed.\n"
        "\n"
        "```python\n"
        "import numpy as np\n"
        "\n"
        "def squelch(x, threshold=0.1):\n"
        "    level = np.abs(x).max()\n"
        "    return (x if level > threshold else None), level\n"
        "```\n"
        "\n"
        "## See Also\n"
        "\n"
        "- **Filter** isolates the channel to gate.\n"
        "- Demodulators such as the **FM Demodulator** usually follow it.\n"
        "- Gain control belongs after it, since **AGC** lifts noise too.\n"
        "- **Cast** converts other types to F32 or CF32."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_DSP_SQUELCH_BLOCK_HH
