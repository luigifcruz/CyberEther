#ifndef JETSTREAM_DOMAINS_DSP_RATIONAL_RESAMPLER_BLOCK_HH
#define JETSTREAM_DOMAINS_DSP_RATIONAL_RESAMPLER_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct RationalResampler : public Block::Config {
    U64 interpolation = 1;
    U64 decimation = 1;
    U64 taps = 0;
    F32 cutoff = 0.9f;

    JST_BLOCK_TYPE(rational_resampler);
    JST_BLOCK_DOMAIN("DSP");
    JST_BLOCK_NODE_SIZE(M);
    JST_BLOCK_PARAMS(interpolation, decimation, taps, cutoff);
    JST_BLOCK_DESCRIPTION(
        "Rational Resampler",
        "Resamples a signal by a rational factor.",
        "# Rational Resampler\n"
        "\n"
        "Changes the sample rate by a ratio of two whole numbers, with a low-pass "
        "filter that removes aliases and images. It usually follows a Filter or a "
        "demodulator, to bring a signal to a rate the next block needs.\n"
        "\n"
        "- **The sample rate scales by the ratio.** The `sampleRate` attribute follows "
        "the new rate.\n"
        "- **Some cycles produce nothing.** When the output length is not a whole "
        "number, downstream blocks occasionally skip a cycle.\n"
        "- **Batches form one stream.** Each channel keeps its own filter history "
        "across batches and buffers.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Input** (in) | F32 or CF32 | Needs a `sampleAxis`, implied for "
        "one-dimensional input. A `batchAxis` and a `channelAxis` are optional. |\n"
        "| **Output** (out) | Same as the input | The input shape with the sample axis "
        "at its length x Interpolation / Decimation, rounded up. Every axis keeps its "
        "role, and `sampleRate` follows the new rate. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **Interpolation** | 1 | 1 or more | Numerator of the rate ratio. Both "
        "factors are reduced by their common divisor. |\n"
        "| **Decimation** | 1 | 1 or more | Denominator of the rate ratio. |\n"
        "| **Taps** | 0 | 0, or odd and at least 2 x the larger reduced factor + 1 | "
        "Filter length at the interpolated rate. Zero picks 64 x the larger reduced "
        "factor + 1. |\n"
        "| **Cutoff** | 0.9 | Above 0, below 1 | Half-amplitude point of the filter, as"
        " a fraction of the lower of the input and output Nyquist frequencies. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "| Goal | Settings | Input | Output |\n"
        "|---|---|---|---|\n"
        "| Demodulated audio to 48 kHz | Interpolation 6, Decimation 25 | 1,000 real "
        "samples at 200 kS/s | 240 real samples at 48 kS/s |\n"
        "| Odd ratio | Interpolation 24, Decimation 125 | 1,024 samples at 250 kS/s | "
        "197 samples at 48 kS/s, with a skipped cycle about every 500 buffers |\n"
        "| Upsample by 1.5 | Interpolation 3, Decimation 2 | 1,024 samples at 250 kS/s "
        "| 1,536 samples at 375 kS/s |\n"
        "| Two channels at once | Interpolation 2, Decimation 3 | 2 channels x 1,024 "
        "samples at 3 MS/s | 2 channels x 683 samples at 2 MS/s |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The rate does not change, but the top of the band is cut | Interpolation and"
        " Decimation are both 1, the default | Set the ratio you need. |\n"
        "| The block reports that Taps must be odd and at least some value | Taps is "
        "even, or too short for the reduced factors | Set Taps to 0 for automatic, or "
        "to an odd value at or above the limit. |\n"
        "| Blocks downstream skip a cycle now and then | The input length x "
        "Interpolation / Decimation is not a whole number | Pick a buffer size that "
        "makes it whole, such as 1,000 samples for 6 / 25. |\n"
        "| The block reports that the rate factors must be positive | Interpolation or "
        "Decimation is 0 | Set both to 1 or more. |\n"
        "| The block reports that the input must be F32 or CF32 | The input has another"
        " type | Convert it with a Cast block. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "The block is a polyphase resampler that computes only the output samples it "
        "keeps, using a Blackman-windowed sinc low-pass with unity gain. Filter history"
        " and phase carry over between buffers, so a stream has no seams, and the "
        "filter adds a fixed delay. The model below covers one buffer from an empty "
        "history.\n"
        "\n"
        "```python\n"
        "import numpy as np\n"
        "from math import ceil, gcd\n"
        "from scipy.signal import firwin, upfirdn\n"
        "\n"
        "def resample(x, interpolation=1, decimation=1, taps=0, cutoff=0.9):\n"
        "    g = gcd(interpolation, decimation)\n"
        "    up, down = interpolation // g, decimation // g\n"
        "    taps = taps or 64 * max(up, down) + 1\n"
        "    h = up * firwin(taps, cutoff / max(up, down), window=\"blackman\")\n"
        "    return upfirdn(h, x, up=up, down=down)[:ceil(len(x) * up / down)]\n"
        "```\n"
        "\n"
        "## See Also\n"
        "\n"
        "- **Decimator** drops the rate by a whole number, with no anti-alias filter.\n"
        "- **Filter** isolates a channel and decimates in one step.\n"
        "- **Audio** resamples on its own for playback.\n"
        "- **Cast** converts other types to F32 or CF32."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_DSP_RATIONAL_RESAMPLER_BLOCK_HH
