#ifndef JETSTREAM_DOMAINS_DSP_RRC_FILTER_BLOCK_HH
#define JETSTREAM_DOMAINS_DSP_RRC_FILTER_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct RrcFilter : public Block::Config {
    F32 symbolRate = 1.0e6f;
    F32 sampleRate = 2.0e6f;
    F32 rollOff = 0.35f;
    U64 taps = 101;

    JST_BLOCK_TYPE(rrc_filter);
    JST_BLOCK_DOMAIN("DSP");
    JST_BLOCK_PARAMS(symbolRate, sampleRate, rollOff, taps);
    JST_BLOCK_DESCRIPTION(
        "RRC Filter",
        "Applies a root raised cosine matched filter to a PSK signal.",
        "# RRC Filter\n"
        "\n"
        "Applies a root raised cosine matched filter to a phase-shift keyed signal, "
        "keeping its length and rate. It usually follows a Filter that isolates the "
        "signal, and feeds a PSK Demodulator.\n"
        "\n"
        "- **Sample Rate is not read from the input.** Set it to match the source.\n"
        "- **Batches form one stream.** Each batch continues the previous one, and each"
        " channel keeps its own history.\n"
        "- **The output lags by half the taps.** The delay is fixed and does not depend"
        " on the signal.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Input** (in) | F32 or CF32 | Needs a `sampleAxis`, implied for "
        "one-dimensional input. A `batchAxis` and a `channelAxis` are optional. The "
        "buffer must be contiguous. |\n"
        "| **Output** (out) | Same as the input | Same shape, with every axis and "
        "attribute kept. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **Sample Rate** | 2 MHz | Above Symbol Rate | Rate of the input. It is not "
        "read from the input, so it must match the source. |\n"
        "| **Symbol Rate** | 1 MHz | Above 0, below Sample Rate | Symbols per second of"
        " the signal. |\n"
        "| **Roll-off Factor** | 0.35 | 0 to 1 | Excess bandwidth, so the signal spans "
        "Symbol Rate x (1 + Roll-off Factor). It must match the transmitter. |\n"
        "| **Taps** | 101 | Odd, 3 or more | Filter length. More taps follow the ideal "
        "pulse more closely and add delay. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "All rows assume a 2 MS/s source.\n"
        "\n"
        "| Goal | Settings | Input | Output |\n"
        "|---|---|---|---|\n"
        "| Two samples per symbol | Defaults | 8,192 complex samples | 8,192 samples, "
        "50 samples late, with taps spanning 50.5 symbols |\n"
        "| Four samples per symbol | Symbol Rate 0.5 MHz | 8,192 complex samples | "
        "8,192 samples, 50 samples late, with taps spanning 25.3 symbols |\n"
        "| Shorter delay | Taps 41 | 8,192 complex samples | 8,192 samples, 20 samples "
        "late, with taps spanning 20.5 symbols |\n"
        "| Batched capture | Defaults | 8 batches x 1,024 complex samples | 8 batches x"
        " 1,024 samples, filtered as one stream |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The block reports that the sample rate must be greater than the symbol rate "
        "| Sample Rate is at or below Symbol Rate | Set Sample Rate to the source rate,"
        " or lower Symbol Rate. |\n"
        "| The constellation smears after the filter | Sample Rate, Symbol Rate, or "
        "Roll-off Factor does not match the signal | Match all three to the source and "
        "the transmitter. |\n"
        "| The level rises after the filter | The taps have unit energy, so the gain at"
        " 0 Hz is the root of the samples per symbol, 1.41 at the defaults | Put an AGC"
        " after the filter when the next block expects a level near 1. |\n"
        "| The block reports that the number of taps must be odd | Taps is even | Add "
        "or remove one tap. |\n"
        "| The block reports that it expects a contiguous tensor | The input is "
        "strided, such as a Slice with Contiguous off | Turn on Contiguous in the "
        "Slice. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "The taps sample the root raised cosine pulse and are scaled to unit energy. A "
        "direct convolution applies them, continuing from the samples of the previous "
        "buffer. The model below covers one buffer without that history.\n"
        "\n"
        "```python\n"
        "import numpy as np\n"
        "\n"
        "def rrc_filter(x, sample_rate=2e6, symbol_rate=1e6, roll_off=0.35, taps=101):\n"
        "    sps, b = sample_rate / symbol_rate, roll_off\n"
        "    t = (np.arange(taps) - (taps - 1) / 2) / sps\n"
        "    edge = np.isclose(np.abs(4 * b * t), 1)\n"
        "    with np.errstate(divide=\"ignore\", invalid=\"ignore\"):\n"
        "        h = np.sin(np.pi * t * (1 - b)) + 4 * b * t * np.cos(np.pi * t * (1 + b))\n"
        "        h /= np.pi * t * (1 - (4 * b * t) ** 2)\n"
        "        q = np.divide(np.pi, 4 * b)\n"
        "        s = (1 + 2 / np.pi) * np.sin(q) + (1 - 2 / np.pi) * np.cos(q)\n"
        "        h[edge] = b / np.sqrt(2) * s\n"
        "    h[t == 0] = 1 + b * (4 / np.pi - 1)\n"
        "    return np.convolve(x, h / np.sqrt(sps))[:len(x)]\n"
        "```\n"
        "\n"
        "## See Also\n"
        "\n"
        "- The filtered signal goes to the **PSK Demodulator**.\n"
        "- **Filter** isolates the signal before this block.\n"
        "- The level returns near 1 with an **AGC**.\n"
        "- **Constellation** shows whether the symbols are clean."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_DSP_RRC_FILTER_BLOCK_HH
