#ifndef JETSTREAM_DOMAINS_DSP_FILTER_TAPS_BLOCK_HH
#define JETSTREAM_DOMAINS_DSP_FILTER_TAPS_BLOCK_HH

#include <vector>

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct FilterTaps : public Block::Config {
    F32 sampleRate = 2.0e6f;
    F32 bandwidth = 1.0e6f;
    std::vector<F32> center = {0.0e6f};
    U64 taps = 101;
    U64 heads = 1;

    JST_BLOCK_TYPE(filter_taps);
    JST_BLOCK_DOMAIN("DSP");
    JST_BLOCK_PARAMS(sampleRate, bandwidth, center, taps, heads);
    JST_BLOCK_DESCRIPTION(
        "Filter Taps",
        "Generates FIR bandpass filter coefficients.",
        "# Filter Taps\n"
        "\n"
        "Generates the coefficients of one or more bandpass filters, one set per head. "
        "It usually feeds the Filter port of a Filter Engine that applies them to a "
        "radio signal.\n"
        "\n"
        "- **One row per head, even for one.** The output always has a `channelAxis` "
        "before the taps.\n"
        "- **The settings travel with the taps.** Filter Engine reads them to decimate "
        "each head.\n"
        "- **Sample Rate is not read from the signal.** Set it to match the source.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Coefficients** (out) | CF32 | Heads x Taps values on a `channelAxis` and a"
        " `sampleAxis`. Sets `sampleRate`, `bandwidth`, and `center` from the settings."
        " |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **Sample Rate** | 2 MHz | Above 0 | Rate of the signal to filter. It must "
        "match the source. |\n"
        "| **Bandwidth** | 1 MHz | Above 0, up to Sample Rate | Passband width between "
        "the half-amplitude points. |\n"
        "| **Heads** | 1 | 1 or more | Number of filters, one row each. |\n"
        "| **Center** | 0 MHz | Within half the Sample Rate | Offset of each passband "
        "from 0 Hz, one value per head. Missing values are 0 MHz, and extra values are "
        "ignored. |\n"
        "| **Taps** | 101 | Odd | Coefficients per head. More taps give a sharper "
        "transition at a higher cost. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "All rows keep the default Sample Rate of 2 MHz.\n"
        "\n"
        "| Goal | Settings | Output |\n"
        "|---|---|---|\n"
        "| One FM station | Bandwidth 0.2 MHz, Center 0.4 MHz | 1 head x 101 taps, "
        "ready to decimate by 10 |\n"
        "| Three FM stations | Heads 3, Bandwidth 0.2 MHz, Center 0, 0.4, -0.4 MHz | 3 "
        "heads x 101 taps |\n"
        "| Taps for power-of-two batches | Bandwidth 0.25 MHz, Taps 97 | 1 head x 97 "
        "taps, which decimate 8,192-sample buffers to 1,024 |\n"
        "| Sharper band edges | Bandwidth 0.2 MHz, Taps 401 | 1 head x 401 taps |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The block reports that the number of taps must be odd | Taps is even | Use "
        "an odd count, such as 101. |\n"
        "| The block reports that a center frequency is out of range | A Center lies "
        "beyond half the Sample Rate | Bring it within half the Sample Rate. |\n"
        "| The block reports that Bandwidth must be between 0 and the sample rate | "
        "Bandwidth is 0 or above Sample Rate | Lower Bandwidth to at most Sample Rate. "
        "|\n"
        "| The wrong signal comes out of the Filter Engine | Sample Rate does not match"
        " the source | Set Sample Rate to the source rate. |\n"
        "| The log warns that Filter Engine bypasses resampling | Sample Rate divided "
        "by Bandwidth is not a whole number, or Taps minus one or the signal length is "
        "not a multiple of it | Pick a Bandwidth that divides Sample Rate evenly, then "
        "adjust Taps or the buffer size. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "Each head is a Blackman-windowed sinc lowpass, shifted to its center by a "
        "complex exponential. The taps are computed once when the block is created, not"
        " for every buffer. The model below covers one head.\n"
        "\n"
        "```python\n"
        "import numpy as np\n"
        "\n"
        "def filter_taps(sample_rate=2e6, bandwidth=1e6, center=0.0, taps=101):\n"
        "    n = np.arange(taps) - (taps - 1) / 2\n"
        "    cutoff = bandwidth / sample_rate\n"
        "    shift = np.exp(2j * np.pi * center / sample_rate * n)\n"
        "    return cutoff * np.sinc(cutoff * n) * np.blackman(taps) * shift\n"
        "```\n"
        "\n"
        "## See Also\n"
        "\n"
        "- **Filter Engine** applies these taps to a signal.\n"
        "- **Filter** combines both blocks in one.\n"
        "- **Slice** picks one head after the engine."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_DSP_FILTER_TAPS_BLOCK_HH
