#ifndef JETSTREAM_DOMAINS_DSP_FILTER_BLOCK_HH
#define JETSTREAM_DOMAINS_DSP_FILTER_BLOCK_HH

#include <vector>

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct Filter : public Block::Config {
    F32 sampleRate = 2.0e6f;
    F32 bandwidth = 1.0e6f;
    std::vector<F32> center = {0.0e6f};
    U64 taps = 101;
    U64 heads = 1;

    JST_BLOCK_TYPE(filter);
    JST_BLOCK_DOMAIN("DSP");
    JST_BLOCK_PARAMS(sampleRate, bandwidth, center, taps, heads);
    JST_BLOCK_DESCRIPTION(
        "Filter",
        "Filters input signal with a FIR bandpass filter.",
        "# Filter\n"
        "\n"
        "Extracts one or more channels from a wideband signal with a bandpass filter. "
        "It usually sits right after a radio source, to isolate one station or to split"
        " a capture into several channels for separate demodulators.\n"
        "\n"
        "- **One channel per head.** Each Center is an offset from the tuned frequency.\n"
        "- **Decimates when the sizes allow.** Each channel then drops to the Bandwidth"
        " rate near 0 Hz, otherwise it stays at its offset.\n"
        "- **Sample Rate is not read from the input.** Set it to match the source.\n"
        "- **Batches form one stream.** Each batch continues the previous one, and the "
        "filter tail carries into the next buffer.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Signal** (in) | F32 or CF32 | Needs a `sampleAxis`, implied for "
        "one-dimensional input. A `batchAxis` is optional and a `channelAxis` is "
        "rejected. A complex signal must be contiguous. |\n"
        "| **Output** (out) | CF32 | The input shape plus a `channelAxis` before the "
        "sample axis, one entry per head. Decimation divides the sample axis and sets "
        "`sampleRate` to Bandwidth. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **Sample Rate** | 2 MHz | Above 0 | Rate of the input. It is not read from "
        "the input, so it must match the source. |\n"
        "| **Bandwidth** | 1 MHz | Above 0, up to Sample Rate | Passband width between "
        "the half-amplitude points. |\n"
        "| **Heads** | 1 | 1 or more | Number of channels extracted in parallel. |\n"
        "| **Center** | 0 MHz | Within half the Sample Rate | Offset of each head from "
        "the tuned frequency, one value per head. Missing values are 0 MHz, and extra "
        "values are ignored. |\n"
        "| **Taps** | 101 | Odd, at most the input length plus one | Filter length. "
        "More taps give a sharper transition at a higher cost. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "All rows assume a 2 MS/s source.\n"
        "\n"
        "| Goal | Settings | Input | Output |\n"
        "|---|---|---|---|\n"
        "| One FM station | Bandwidth 0.2 MHz, Center 0.4 MHz | 8,190 complex samples |"
        " 1 head x 819 samples at 200 kS/s |\n"
        "| Three FM stations | Heads 3, Bandwidth 0.2 MHz, Center 0, 0.4, -0.4 MHz | "
        "8,190 complex samples | 3 heads x 819 samples at 200 kS/s |\n"
        "| Power-of-two buffers | Bandwidth 0.25 MHz, Taps 97 | 8,192 complex samples |"
        " 1 head x 1,024 samples at 250 kS/s |\n"
        "| Filter only | Bandwidth 0.3 MHz | 8,192 complex samples | 1 head x 8,192 "
        "samples at 2 MS/s |\n"
        "| Real input | Bandwidth 0.2 MHz | 8,190 real samples | 1 head x 819 complex "
        "samples at 200 kS/s |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The log warns that resampling is bypassed | Sample Rate divided by Bandwidth"
        " is not a whole number, or the input length or Taps minus one is not a "
        "multiple of it | Pick a Bandwidth that divides Sample Rate evenly, then adjust"
        " the buffer size or Taps, as in the Power-of-two buffers recipe. |\n"
        "| The wrong signal comes out, or none at all | Sample Rate does not match the "
        "source | Set Sample Rate to the source rate. |\n"
        "| The log warns that the output is shifted by a small frequency | A Center is "
        "not a multiple of the bin size, which is Sample Rate divided by the input "
        "length plus Taps minus one | Move that Center to a multiple of the bin size. |\n"
        "| The block reports that the signal already has a channel axis | The input "
        "already has a `channelAxis` | Pick one channel with a Slice block first. |\n"
        "| The block reports that it expects a contiguous tensor | The complex signal "
        "is strided, such as a Slice with Contiguous off | Turn on Contiguous in the "
        "Slice. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "The taps are a Blackman-windowed sinc, shifted to each center and applied by "
        "FFT overlap-add convolution, with one forward transform shared by every head. "
        "Decimation folds the filtered spectrum before the inverse FFT, which matches "
        "mixing each channel down and dropping the extra samples. The model below "
        "covers a single head without the decimation conditions.\n"
        "\n"
        "```python\n"
        "import numpy as np\n"
        "\n"
        "def filter_head(x, sample_rate=2e6, bandwidth=1e6, center=0.0, taps=101):\n"
        "    n = np.arange(taps) - (taps - 1) / 2\n"
        "    cutoff = bandwidth / sample_rate\n"
        "    shift = np.exp(2j * np.pi * center / sample_rate * n)\n"
        "    h = cutoff * np.sinc(cutoff * n) * np.blackman(taps) * shift\n"
        "    y = np.convolve(x, h)[:len(x)]\n"
        "    bins = len(x) + taps - 1\n"
        "    k = center / sample_rate * bins\n"
        "    mixed = np.sign(k) * np.floor(abs(k) + 0.5) / bins\n"
        "    down = np.exp(-2j * np.pi * mixed * np.arange(len(y)))\n"
        "    return (y * down)[::int(sample_rate // bandwidth)]\n"
        "```\n"
        "\n"
        "## See Also\n"
        "\n"
        "- **Filter Taps** and **Filter Engine** split this block in two.\n"
        "- **Slice** picks one head for single-channel blocks.\n"
        "- **Rational Resampler** handles non-integer rate changes."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_DSP_FILTER_BLOCK_HH
