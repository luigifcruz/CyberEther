#ifndef JETSTREAM_DOMAINS_DSP_AGC_BLOCK_HH
#define JETSTREAM_DOMAINS_DSP_AGC_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct Agc : public Block::Config {
    U64 tileSize = 1024;
    F32 reference = 1.0f;
    F32 epsilon = 1e-12f;
    F32 minGain = 0.01f;
    F32 maxGain = 100.0f;
    F32 maxGainChange = 4.0f;

    JST_BLOCK_TYPE(agc);
    JST_BLOCK_DOMAIN("DSP");
    JST_BLOCK_NODE_SIZE(XS);
    JST_BLOCK_PARAMS(tileSize, reference, epsilon, minGain, maxGain,
                     maxGainChange);
    JST_BLOCK_DESCRIPTION(
        "AGC",
        "Normalizes a signal to a target RMS level.",
        "# AGC\n"
        "\n"
        "Scales a signal toward a target RMS level, with one gain per tile of samples "
        "that ramps smoothly into the next. It usually follows a Filter or a "
        "demodulator, and feeds a block that needs a steady level, such as Audio or "
        "Constellation.\n"
        "\n"
        "- **Quiet input is lifted up to Max Gain.** Noise between transmissions rises "
        "by up to 100 times at the defaults.\n"
        "- **Each lane is independent.** Every batch and channel is processed "
        "separately.\n"
        "- **Gain restarts with every buffer.** Max Gain Change does not limit the step"
        " between one buffer and the next.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Input** (in) | F32 or CF32 | Needs a `sampleAxis`, implied for "
        "one-dimensional input. The buffer must be contiguous. |\n"
        "| **Output** (out) | Same as the input | Same shape, with every axis and "
        "attribute kept. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **Tile Size** | 1,024 samples | 1 or more | Samples per RMS estimate, about "
        "the sample rate times the response time. Shorter tiles react faster but also "
        "follow the modulation. |\n"
        "| **Reference** | 1 | Above 0 | Target RMS level of the output. |\n"
        "| **Epsilon** | 1e-12 | Above 0 | Floor added to the mean power, so silence "
        "asks for a finite gain. |\n"
        "| **Min Gain** | 0.01 | Above 0 | Lowest gain a tile uses. |\n"
        "| **Max Gain** | 100 | Min Gain or more | Highest gain a tile uses. |\n"
        "| **Max Gain Change** | 4 | 1 or more | Largest gain ratio between neighboring"
        " tiles. Bigger level jumps take several tiles. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "All rows use buffers of 8,192 samples, eight tiles at the default Tile Size.\n"
        "\n"
        "| Goal | Settings | Input | Output |\n"
        "|---|---|---|---|\n"
        "| Level a weak signal | Defaults | Signal at RMS 0.05 | Scaled by 20 to RMS 1 "
        "|\n"
        "| Keep noise down | Max Gain 10 | Noise at RMS 0.001 | Scaled by 10 to RMS "
        "0.01 |\n"
        "| Level each channel | Defaults | 2 channels at RMS 0.1 and 2 | Both channels "
        "at RMS 1 |\n"
        "| Burst after silence | Defaults | Complex tone at 0.01, then 1 from sample "
        "4,096 | Overshoots to 25 at sample 4,096, back to 1 from sample 7,168 |\n"
        "| Fast attack | Max Gain Change 100 | The same burst | Amplitude 1 from sample"
        " 4,096 on |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| Noise turns into loud hiss when no signal is present | Quiet input gets up "
        "to Max Gain, 100 by default | Lower Max Gain, or gate the input with a Squelch"
        " first. |\n"
        "| A burst after a quiet spell comes out far above Reference | Max Gain Change "
        "slows how fast the gain falls from Max Gain | Raise Max Gain Change, or lower "
        "Max Gain. |\n"
        "| Envelopes such as AM audio come out flattened | Tile Size is shorter than "
        "the modulation, so the gain follows it | Raise Tile Size to span several "
        "cycles of the slowest modulation. |\n"
        "| The block reports an unsupported data type | The input is not F32 or CF32 | "
        "Convert it with a Cast block. |\n"
        "| The block reports that it expects a contiguous tensor | The input is "
        "strided, such as a Slice with Contiguous off | Turn on Contiguous in the "
        "Slice. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "Each tile gets the gain that brings its RMS to the reference, limited by the "
        "gain bounds and the change allowed from the previous tile. Within a tile the "
        "gain ramps linearly toward the next tile's gain, and samples that would "
        "overflow are clipped with their phase kept. The model below covers one lane of"
        " one buffer without the clipping.\n"
        "\n"
        "```python\n"
        "import numpy as np\n"
        "\n"
        "def agc(x, tile_size=1024, reference=1.0, epsilon=1e-12, min_gain=0.01,\n"
        "        max_gain=100.0, max_gain_change=4.0):\n"
        "    tiles = [x[i:i + tile_size] for i in range(0, len(x), tile_size)]\n"
        "    g = [reference / np.sqrt(np.mean(np.abs(t) ** 2) + epsilon) for t in tiles]\n"
        "    g = list(np.clip(g, min_gain, max_gain))\n"
        "    for k in range(1, len(g)):\n"
        "        low = max(min_gain, g[k - 1] / max_gain_change)\n"
        "        g[k] = np.clip(g[k], low, min(max_gain, g[k - 1] * max_gain_change))\n"
        "    g.append(g[-1])\n"
        "    ramps = [np.linspace(g[k], g[k + 1], len(t), endpoint=False)\n"
        "             for k, t in enumerate(tiles)]\n"
        "    return x * np.concatenate(ramps)\n"
        "```\n"
        "\n"
        "## See Also\n"
        "\n"
        "- **Squelch** goes first, so noise is not lifted.\n"
        "- **Multiply Constant** applies a fixed gain instead.\n"
        "- **Cast** converts other types to F32 or CF32."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_DSP_AGC_BLOCK_HH
