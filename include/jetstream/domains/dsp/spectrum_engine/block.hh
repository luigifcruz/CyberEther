#ifndef JETSTREAM_DOMAINS_DSP_SPECTRUM_ENGINE_BLOCK_HH
#define JETSTREAM_DOMAINS_DSP_SPECTRUM_ENGINE_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct SpectrumEngine : public Block::Config {
    bool enableAgc = false;
    bool enableScale = false;
    F32 rangeMin = -120.0f;
    F32 rangeMax = 0.0f;

    JST_BLOCK_TYPE(spectrum_engine);
    JST_BLOCK_DOMAIN("DSP");
    JST_BLOCK_NODE_SIZE(M);
    JST_BLOCK_PARAMS(enableAgc, enableScale, rangeMin, rangeMax);
    JST_BLOCK_DESCRIPTION(
        "Spectrum Engine",
        "Computes spectra with windowing, FFT, and optional scaling.",
        "# Spectrum Engine\n"
        "\n"
        "Computes the spectrum of each buffer in decibels, with optional gain control "
        "and scaling. It usually follows a radio source, and feeds a Waterfall or a "
        "Lineplot.\n"
        "\n"
        "- **Each lane is independent.** Every batch and channel gets its own spectrum.\n"
        "- **The tuned frequency sits in the middle bin.** Negative offsets fill the "
        "left half.\n"
        "- **A full-scale tone reads about -7.5 dBFS.** The window loss is not "
        "corrected.\n"
        "- **Views need Enable Scale on.** Waterfall and Lineplot expect values from 0 "
        "to 1.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Input** (in) | F32 or CF32. Real input gives a mirrored spectrum. | Needs "
        "a `sampleAxis`, implied for one-dimensional input. A `batchAxis` and a "
        "`channelAxis` are optional. |\n"
        "| **Output** (out) | F32 | Same shape, with every axis and attribute kept. "
        "Each sample becomes one bin. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **Enable AGC** | Off | On, Off | Scales each spectrum so its RMS level reads"
        " -78.3 dBFS at 8,192 bins, with at most 40 dB of gain or loss. Not available "
        "on CUDA. |\n"
        "| **Enable Scale** | Off | On, Off | Maps Range Min to 0 and Range Max to 1, "
        "without clipping. |\n"
        "| **Range Min** | -120 dBFS | Any, slider -300 to 0 | Level mapped to 0. "
        "Hidden while Enable Scale is off. |\n"
        "| **Range Max** | 0 dBFS | Any, slider -300 to 0 | Level mapped to 1. Hidden "
        "while Enable Scale is off. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "All rows assume a 2 MS/s source.\n"
        "\n"
        "| Goal | Settings | Input | Output |\n"
        "|---|---|---|---|\n"
        "| Feed a Waterfall | Enable Scale on | 8,192 samples | 8,192 bins of 244 Hz, "
        "from 0 to 1 |\n"
        "| Batched spectra | Defaults | 8 batches x 1,024 samples | 8 batches x 1,024 "
        "bins of 1.95 kHz, in dBFS |\n"
        "| Steady level | Enable AGC on | 8,192 samples | 8,192 bins with the RMS level"
        " at -78.3 dBFS |\n"
        "| Real signal | Defaults | 4,096 real samples | 4,096 bins, mirrored around "
        "the middle |\n"
        "| One spectrum per head | Enable Scale on | 3 heads x 819 samples | 3 heads x "
        "819 bins, from 0 to 1 |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| A Waterfall fed by this block is one flat color | Enable Scale is off, so "
        "values sit in dBFS | Turn on Enable Scale and set the range around the signal."
        " |\n"
        "| The block reports that the input must have data type F32 or CF32 | The input"
        " is an integer type, such as raw CI16 samples | Convert it to CF32 with a Cast"
        " block. |\n"
        "| The block reports that the input signal axis metadata is invalid | A "
        "multi-dimensional input has no `sampleAxis` | Assign roles with a Signal Axes "
        "block. |\n"
        "| Every bin reads 0.5 | Range Min equals Range Max | Set Range Min below Range"
        " Max. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "Each lane gets a Blackman window, modulated so the transform comes out "
        "centered, then the magnitude is divided by the length and converted to "
        "decibels. Enable AGC applies one gain per spectrum, taken from its RMS level. "
        "The model below covers one lane, with Enable Scale as a second step.\n"
        "\n"
        "```python\n"
        "import numpy as np\n"
        "\n"
        "def spectrum(x, enable_agc=False):\n"
        "    s = np.fft.fftshift(np.fft.fft(x * np.blackman(len(x))))\n"
        "    if enable_agc:\n"
        "        rms = np.sqrt(np.mean(np.abs(s) ** 2) + 1e-12)\n"
        "        s = s * np.clip(1 / rms, 0.01, 100)\n"
        "    return 20 * np.log10(np.abs(s) / len(x))\n"
        "\n"
        "def scale(db, range_min=-120.0, range_max=0.0):\n"
        "    lo, hi = sorted((range_min, range_max))\n"
        "    if lo == hi:\n"
        "        return np.full_like(db, 0.5)\n"
        "    y = (db - lo) / (hi - lo)\n"
        "    return np.where(np.isfinite(y), y, (y > 0).astype(float))\n"
        "```\n"
        "\n"
        "## See Also\n"
        "\n"
        "- **Waterfall** draws the scaled spectrum as a history.\n"
        "- **Lineplot** draws the scaled spectrum as a trace.\n"
        "- **Spectrum Analyzer** computes and draws a spectrum in one block.\n"
        "- **Cast** converts integer samples to CF32."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_DSP_SPECTRUM_ENGINE_BLOCK_HH
