#ifndef JETSTREAM_DOMAINS_DSP_FM_BLOCK_HH
#define JETSTREAM_DOMAINS_DSP_FM_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct FM : public Block::Config {
    std::string mode = "narrow";
    std::string deemphasis = "none";
    F32 sampleRate = 240e3f;

    JST_BLOCK_TYPE(fm);
    JST_BLOCK_DOMAIN("DSP");
    JST_BLOCK_NODE_SIZE(M);
    JST_BLOCK_PARAMS(mode, deemphasis, sampleRate);
    JST_BLOCK_DESCRIPTION(
        "FM Demodulator",
        "Demodulates a frequency modulated signal.",
        "# FM Demodulator\n"
        "\n"
        "Turns frequency-modulated samples into mono or stereo audio. It usually "
        "follows a Filter that isolates one station, and feeds an Audio block that "
        "plays the result.\n"
        "\n"
        "- **Wideband is stereo.** Left and right come out on a new `channelAxis`.\n"
        "- **Sample Rate is not read from the input.** Set it to match the block "
        "upstream.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Input** (in) | CF32 | Needs a contiguous buffer with a `sampleAxis`, "
        "implied for one-dimensional input. A `batchAxis` is optional. A `channelAxis` "
        "is accepted in Narrowband only. |\n"
        "| **Output** (out) | F32 | Same shape as the input, with `frequency` set to 0."
        " Wideband appends a `channelAxis` of 2, left then right. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **Mode** | Narrowband | Narrowband, Wideband | Narrowband outputs mono, full"
        " scale at 100 kHz of deviation. Wideband decodes stereo, full scale at 75 kHz."
        " |\n"
        "| **De-emphasis** | None | None, 50 us [Global], 75 us [USA] | Rolls off the "
        "treble that broadcast stations boost. |\n"
        "| **Sample Rate** | 0.24 MHz | Above 0, up to 20 MHz. At least 0.2 MHz in "
        "Wideband. | Rate of the input. It is not read from the input, so it must match"
        " the block upstream. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "| Goal | Settings | Input | Output |\n"
        "|---|---|---|---|\n"
        "| Stereo broadcast | Wideband, 75 us, Sample Rate 0.2 MHz | 819 complex "
        "samples at 200 kS/s | 819 samples x 2 channels (left, right) |\n"
        "| Mono broadcast | Narrowband, 75 us, Sample Rate 0.2 MHz | 819 complex "
        "samples at 200 kS/s | 819 real samples |\n"
        "| Three stations at once | Narrowband, Sample Rate 0.2 MHz | 3 heads x 819 "
        "complex samples, from a Filter | 3 heads x 819 real samples |\n"
        "| Batched stereo | Wideband, Sample Rate 0.24 MHz | 8 batches x 1,000 complex "
        "samples | 8 batches x 1,000 samples x 2 channels |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The block reports that Wideband needs at least 200 kHz | Sample Rate is "
        "below 0.2 MHz | Raise the rate upstream, or switch to Narrowband. |\n"
        "| The block reports that Wideband does not support channelized input | The "
        "input has a `channelAxis`, such as a multi-head Filter output | Pick one head "
        "with a Slice block, or switch to Narrowband. |\n"
        "| Stereo never separates, or the level and treble sound wrong | Sample Rate "
        "does not match the input | Set Sample Rate to the rate of the block upstream. "
        "|\n"
        "| Narrowband audio is very quiet | The output is scaled for 100 kHz of "
        "deviation | Add gain downstream with Multiply Constant. |\n"
        "| The block reports that the input must be complex | The input is not CF32 | "
        "Feed it complex samples from a radio or a Filter. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "The demodulator measures the phase change between consecutive samples and "
        "scales it by a fixed reference deviation for each mode. Wideband also tracks "
        "the stereo pilot to split left and right, and band-limits both channels to the"
        " audio range. The model below covers Narrowband for one buffer.\n"
        "\n"
        "```python\n"
        "import numpy as np\n"
        "from scipy.signal import lfilter\n"
        "\n"
        "def fm_demodulate(x, sample_rate=240e3, deviation=100e3, deemphasis=None):\n"
        "    previous = np.concatenate(([x[0]], x[:-1]))\n"
        "    y = np.angle(x * np.conj(previous)) * sample_rate / (2 * np.pi * deviation)\n"
        "    if deemphasis:\n"
        "        alpha = 1 - np.exp(-1 / (sample_rate * deemphasis))\n"
        "        y = lfilter([alpha], [1, alpha - 1], y)\n"
        "    return y\n"
        "```\n"
        "\n"
        "## See Also\n"
        "\n"
        "- **Filter** isolates a station before demodulation.\n"
        "- **Audio** plays the result.\n"
        "- Amplitude-modulated signals go to the **AM Demodulator**.\n"
        "- **Slice** picks one channel from a multi-head Filter."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_DSP_FM_BLOCK_HH
