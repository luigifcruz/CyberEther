#ifndef JETSTREAM_DOMAINS_DSP_PSK_DEMOD_BLOCK_HH
#define JETSTREAM_DOMAINS_DSP_PSK_DEMOD_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct PskDemod : public Block::Config {
    std::string pskType = "qpsk";
    F32 sampleRate = 2000000.0f;
    F32 symbolRate = 1000000.0f;
    F32 frequencyLoopBandwidth = 0.05f;
    F32 timingLoopBandwidth = 0.05f;
    F32 dampingFactor = 0.707f;

    JST_BLOCK_TYPE(psk_demod);
    JST_BLOCK_DOMAIN("DSP");
    JST_BLOCK_NODE_SIZE(M);
    JST_BLOCK_PARAMS(pskType, sampleRate, symbolRate, frequencyLoopBandwidth,
                     timingLoopBandwidth, dampingFactor);
    JST_BLOCK_DESCRIPTION(
        "PSK Demodulator",
        "Demodulates PSK signals with carrier and timing recovery.",
        "# PSK Demodulator\n"
        "\n"
        "Recovers the carrier and the symbol timing of a phase-shift keyed signal, and "
        "outputs one soft symbol per symbol period. It usually follows a Filter that "
        "isolates the signal and an RRC Filter, and feeds a Constellation view.\n"
        "\n"
        "- **One output sample per symbol.** The `sampleRate` attribute becomes Symbol "
        "Rate.\n"
        "- **Sample Rate is not read from the input.** Set it to match the source.\n"
        "- **Batches form one stream.** Loop state carries across batches and buffers, "
        "and each channel has its own loops.\n"
        "- **Loops expect a level near 1.** Put an AGC upstream, since the loop gains "
        "scale with the signal level.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Input** (in) | CF32 | Needs a `sampleAxis`, implied for one-dimensional "
        "input. A `batchAxis` and a `channelAxis` are optional. The buffer must be "
        "contiguous. |\n"
        "| **Output** (out) | CF32 | The input shape with the sample axis cut to the "
        "symbol count, rounded up. Sets `sampleRate` to Symbol Rate. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **PSK Type** | QPSK | BPSK, QPSK, 8-PSK | Constellation the loops lock to. |\n"
        "| **Sample Rate** | 2 MHz | Above 0 | Rate of the input. It is not read from "
        "the input, so it must match the source. |\n"
        "| **Symbol Rate** | 1 MHz | Above 0, up to half the Sample Rate | Symbols per "
        "second, which sets the output length and rate. |\n"
        "| **Freq Loop BW** | 0.05 | Above 0 and below 1, slider 0.001 to 0.2 | Carrier"
        " loop bandwidth per symbol. Wider locks faster but adds jitter. |\n"
        "| **Timing Loop BW** | 0.05 | Above 0 and below 1, slider 0.001 to 0.2 | "
        "Timing loop bandwidth per symbol. |\n"
        "| **Damping Factor** | 0.707 | Above 0, slider 0.1 to 2 | Damping of both "
        "loops. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "All rows assume a 2 MS/s source.\n"
        "\n"
        "| Goal | Settings | Input | Output |\n"
        "|---|---|---|---|\n"
        "| Defaults | Defaults | 8,192 samples | 4,096 symbols at 1 MS/s |\n"
        "| Four samples per symbol | Symbol Rate 0.5 MHz | 8,192 samples | 2,048 "
        "symbols at 500 kS/s |\n"
        "| Slow BPSK | Symbol Rate 0.125 MHz, BPSK | 8,192 samples | 512 symbols at 125"
        " kS/s |\n"
        "| Batched 8-PSK | 8-PSK, Symbol Rate 0.5 MHz | 8 batches x 1,024 samples | 8 "
        "batches x 256 symbols, one continuous stream |\n"
        "| Uneven ratio | Symbol Rate 0.3 MHz | 8,192 samples | 1,229 symbols at 300 "
        "kS/s, with a buffer skipped now and then |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The block reports that samples per symbol must be at least 2 | Symbol Rate "
        "is above half the Sample Rate | Lower Symbol Rate, or raise the source rate. |\n"
        "| The constellation spins or smears | Sample Rate or Symbol Rate does not "
        "match the signal, or the level is far from 1 | Match both rates to the signal,"
        " and add an AGC upstream. |\n"
        "| The block reports that the pending symbol capacity was exceeded | The signal"
        " runs faster than Symbol Rate, so symbols pile up | Set Symbol Rate to the "
        "exact rate of the signal. |\n"
        "| Downstream blocks skip a buffer now and then | The buffer does not hold a "
        "whole number of symbols, so the block waits for a full output | Pick a buffer "
        "size that holds a whole number of symbols. |\n"
        "| The block reports that the input must be complex | The input is not CF32 | "
        "Feed it complex samples from a radio or a Filter. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "A Costas loop removes the carrier offset and a Mueller and Muller detector "
        "steers a linear interpolator to the symbol centers. The output keeps the "
        "corrected samples without hard decisions or matched filtering. The model below"
        " covers the QPSK carrier loop at fixed symbol timing for one buffer.\n"
        "\n"
        "```python\n"
        "import numpy as np\n"
        "\n"
        "def psk_demod(x, sample_rate=2e6, symbol_rate=1e6,\n"
        "              frequency_loop_bandwidth=0.05, damping_factor=0.707):\n"
        "    bw, z = frequency_loop_bandwidth, damping_factor\n"
        "    alpha = 4 * z * bw / (1 + 2 * z * bw + bw * bw)\n"
        "    beta = 4 * bw * bw / (1 + 2 * z * bw + bw * bw)\n"
        "    y = x[np.arange(0, len(x) - 1, sample_rate / symbol_rate).astype(int)]\n"
        "    phase, freq = 0.0, 0.0\n"
        "    for i in range(len(y)):\n"
        "        y[i] *= np.exp(-1j * phase)\n"
        "        d = np.where(y[i].real > 0, 1, -1) + 1j * np.where(y[i].imag > 0, 1, -1)\n"
        "        err = np.clip(np.imag(y[i] * np.conj(d)), -1, 1)\n"
        "        freq = np.clip(freq + beta * err, -np.pi, np.pi)\n"
        "        phase += freq + alpha * err\n"
        "    return y\n"
        "```\n"
        "\n"
        "## See Also\n"
        "\n"
        "- Pulse shaping is matched by the **RRC Filter**.\n"
        "- Levels are evened out by the **AGC**.\n"
        "- **Constellation** shows the soft symbols.\n"
        "- **Filter** isolates the signal and lowers the rate."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_DSP_PSK_DEMOD_BLOCK_HH
