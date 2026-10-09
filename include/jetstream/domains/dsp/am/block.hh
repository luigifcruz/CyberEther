#ifndef JETSTREAM_DOMAINS_DSP_AM_BLOCK_HH
#define JETSTREAM_DOMAINS_DSP_AM_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct AM : public Block::Config {
    F32 sampleRate = 240e3f;
    F32 dcAlpha = 0.995f;

    JST_BLOCK_TYPE(am);
    JST_BLOCK_DOMAIN("DSP");
    JST_BLOCK_NODE_SIZE(M);
    JST_BLOCK_PARAMS(sampleRate, dcAlpha);
    JST_BLOCK_DESCRIPTION(
        "AM Demodulator",
        "Demodulates an amplitude modulated signal.",
        "# AM Demodulator\n"
        "\n"
        "Recovers the audio of an amplitude-modulated signal by following its envelope "
        "and removing the carrier level. It usually follows a Filter that isolates one "
        "station, and feeds an Audio block that plays the result.\n"
        "\n"
        "- **Sample Rate changes nothing.** The output never depends on it, and the "
        "input `sampleRate` passes through.\n"
        "- **The carrier offset does not matter.** The envelope is the same wherever "
        "the carrier sits in the band.\n"
        "- **Batches form one stream.** Each channel keeps its filter state across "
        "batches and buffers.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Input** (in) | CF32 | Needs a `sampleAxis`, implied for one-dimensional "
        "input. A `batchAxis` and a `channelAxis` are optional. The buffer must be "
        "contiguous. |\n"
        "| **Output** (out) | F32 | Same shape, with every axis and attribute kept and "
        "`frequency` set to 0. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **Sample Rate** | 0.24 MHz | Above 0 | Not used by the demodulator. |\n"
        "| **DC Alpha** | 0.995 | 0 up to but not including 1, slider 0.9 to 0.999 | "
        "Pole of the DC blocker. Higher keeps more bass, with the corner near 190 Hz at"
        " 240 kS/s for 0.995. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "All rows assume a 240 kS/s input.\n"
        "\n"
        "| Goal | Settings | Input | Output |\n"
        "|---|---|---|---|\n"
        "| Broadcast station | Defaults | 8,192 complex samples | 8,192 real samples at"
        " 240 kS/s, bass cut below about 190 Hz |\n"
        "| Keep more bass | DC Alpha 0.999 | 8,192 complex samples | 8,192 real "
        "samples, bass cut below about 38 Hz |\n"
        "| Three stations at once | Defaults | 3 heads x 819 complex samples, from a "
        "Filter | 3 heads x 819 real samples |\n"
        "| Batched rows | Defaults | 8 batches x 1,000 complex samples | 8 batches x "
        "1,000 real samples, as one stream per channel |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The block reports that the input must be complex | The input is not CF32 | "
        "Feed it complex samples from a radio or a Filter. |\n"
        "| The block reports that the input needs valid signal axis metadata | A "
        "multi-dimensional input has no `sampleAxis` | Mark the axes with a Signal Axes"
        " block. |\n"
        "| The block reports that it expects a contiguous tensor | The input is "
        "strided, such as a Slice with Contiguous off | Turn on Contiguous in the "
        "Slice. |\n"
        "| Audio is distorted or carries other stations | More than one signal reaches "
        "the envelope | Isolate the station with a Filter first. |\n"
        "| Volume rises and falls with signal strength | The output follows the carrier"
        " level, with no gain control | Add an AGC after the block. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "The block takes the magnitude of each complex sample as the envelope. A "
        "one-pole DC blocker then removes the steady carrier level, starting from rest "
        "when the block is created. The model below covers one buffer from rest.\n"
        "\n"
        "```python\n"
        "import numpy as np\n"
        "from scipy.signal import lfilter\n"
        "\n"
        "def am_demodulate(x, dc_alpha=0.995):\n"
        "    return lfilter([1, -1], [1, -dc_alpha], np.abs(x))\n"
        "```\n"
        "\n"
        "## See Also\n"
        "\n"
        "- **Filter** isolates a station before demodulation.\n"
        "- **Audio** plays the result.\n"
        "- Fading stations even out with an **AGC** after it.\n"
        "- Frequency-modulated signals go to the **FM Demodulator**."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_DSP_AM_BLOCK_HH
