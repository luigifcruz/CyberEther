#ifndef JETSTREAM_DOMAINS_VISUALIZATION_SPECTRUM_ANALYZER_BLOCK_HH
#define JETSTREAM_DOMAINS_VISUALIZATION_SPECTRUM_ANALYZER_BLOCK_HH

#include <string>
#include <vector>

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct SpectrumAnalyzer : public Block::Config {
    std::string mode = "lineplot_waterfall";
    U64 lineplotAveraging = 1;
    U64 waterfallAveraging = 1;
    bool maxHold = false;
    bool fill = true;
    F32 rangeMin = -100.0f;
    F32 rangeMax = 0.0f;
    U64 waterfallHeight = 1024;
    std::string colormap = "turbo";
    F32 splitRatio = 0.5f;
    std::vector<F32> markers;
    std::vector<U64> pins;
    std::string xLabel = "Frequency (MHz)";
    std::string amplitudeLabel = "Amplitude (dBFS)";
    std::string waterfallLabel = "Time";

    JST_BLOCK_TYPE(spectrum_analyzer);
    JST_BLOCK_DOMAIN("Visualization");
    JST_BLOCK_NODE_SIZE(L);
    JST_BLOCK_PARAMS(mode, lineplotAveraging, waterfallAveraging, maxHold,
                     fill, rangeMin, rangeMax, waterfallHeight, colormap,
                     splitRatio, markers, pins, xLabel, amplitudeLabel, waterfallLabel);
    JST_BLOCK_DESCRIPTION(
        "Spectrum Analyzer",
        "Shows the spectrum of complex samples as a trace and a waterfall.",
        "# Spectrum Analyzer\n"
        "\n"
        "Computes the spectrum of complex samples and draws it as a live trace, a "
        "scrolling waterfall, or both. It usually sits right after a radio source, to "
        "watch the captured band.\n"
        "\n"
        "- **Takes raw samples, not a spectrum.** The transform has one bin per input "
        "sample.\n"
        "- **Each batch is one spectrum.** The trace averages the batches, and the "
        "waterfall adds one row per batch.\n"
        "- **Controls work in a detached window.** Click the expand icon on the view to"
        " open it.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Input** (in) | CF32 | Needs a `sampleAxis`, implied for one-dimensional "
        "input. A `batchAxis` is optional, and no other axis is allowed. The axis shows"
        " MHz when `frequency` and `sampleRate` are set. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **Mode** | Lineplot + Waterfall | Lineplot + Waterfall, Spectrum, Waterfall,"
        " 3D Waterfall | Views drawn. The 3D option draws the history as a surface, "
        "without markers. |\n"
        "| **Colormap** | Turbo | Grayscale, Turbo, Viridis, Inferno, Magma, Plasma, "
        "Jet | Palette from Range Min to Range Max. Hidden for Spectrum. |\n"
        "| **Range Min** | -100 dBFS | Any, slider -300 to 0 | Level at the bottom of "
        "the trace and the colors. |\n"
        "| **Range Max** | 0 dBFS | Any, slider -300 to 0 | Level at the top of the "
        "trace and the colors. |\n"
        "| **Lineplot Averaging** | 1 | 1 or more, slider up to 256 | Smooths the trace"
        " with an exponential average over about this many buffers. Averaging in dB "
        "reads noise about 2.51 dB below its mean power. Hidden for Waterfall and 3D "
        "Waterfall. |\n"
        "| **Waterfall Averaging** | 1 | 1 or more, slider up to 64 | Consecutive "
        "spectra averaged into each row. Hidden for Spectrum. |\n"
        "| **Max Hold** | Off | On, Off | Adds a trace of the highest level seen. "
        "Hidden for Waterfall and 3D Waterfall. |\n"
        "| **Waterfall Height** | 1,024 rows | 1 to 8,192, at least 2 for 3D Waterfall "
        "| Rows of history. Hidden for Spectrum. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "All rows assume a 2 MS/s source.\n"
        "\n"
        "| Goal | Settings | Input | Output |\n"
        "|---|---|---|---|\n"
        "| Live band view | Defaults | 8,192 samples per buffer | 8,192 bins and 1,024 "
        "rows covering 4.2 seconds |\n"
        "| Steady trace with peaks | Spectrum, Lineplot Averaging 16, Max Hold on | "
        "8,192 samples per buffer | A trace averaged over about 66 ms, plus the peak "
        "trace |\n"
        "| Long history | Waterfall, Waterfall Averaging 8 | 8,192 samples per buffer |"
        " 1,024 rows covering 33.6 seconds |\n"
        "| Batched capture | Defaults | 8 batches x 1,024 samples | 1,024 bins, 8 "
        "spectra averaged per trace, and 1,024 rows covering 0.52 seconds |\n"
        "| Surface view | 3D Waterfall, Waterfall Height 256 | 8,192 samples per buffer"
        " | A surface of 256 rows covering 1.05 seconds |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The trace hugs an edge, or the waterfall is one color | Signal levels sit "
        "outside Range Min to Range Max | Set the range from just below the noise floor"
        " to just above the strongest signal. |\n"
        "| Zoom and markers do nothing | The view is shown inside the node, which takes"
        " no input | Open it in its own window with the expand icon. |\n"
        "| The view stops updating | Space froze it | Press Space again in its window. "
        "|\n"
        "| The block reports that the input must have data type CF32 | The input is "
        "real, or already a spectrum | Feed it complex samples from the source. |\n"
        "| The block reports that channel inputs are not supported | The input has a "
        "`channelAxis`, such as one from a Filter | Pick one head with a Slice block. |\n"
        "\n"
        "## Controls\n"
        "\n"
        "| Action | Input |\n"
        "|---|---|\n"
        "| Zoom the frequency axis around the cursor, up to 10x | Scroll |\n"
        "| Pan while zoomed | Drag |\n"
        "| Reset zoom and pan | Right Click |\n"
        "| Freeze or resume the view | Space |\n"
        "| Place a marker, or remove the one under the cursor | Shift+Click |\n"
        "| Clear every marker | Shift+Right Click |\n"
        "| Move a marker | Drag its line |\n"
        "| Show the distances to neighboring markers | Hover a marker tag, click it to "
        "pin |\n"
        "| Unpin every tag | Escape |\n"
        "| Resize the trace and waterfall | Drag the border between them. Lineplot + "
        "Waterfall only. |\n"
        "| Orbit around the surface | Drag in 3D Waterfall |\n"
        "| Pan the view | Right Drag in 3D Waterfall |\n"
        "| Move closer or farther | Scroll in 3D Waterfall |\n"
        "| Return to the home view | Right Click in 3D Waterfall |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "Each batch gets a Blackman window and an FFT centered on the tuned frequency, "
        "then the magnitude is converted to dBFS. The trace keeps an exponential "
        "average, the waterfall keeps a ring buffer of rows, and both compress levels "
        "past the range softly instead of clipping.\n"
        "\n"
        "## See Also\n"
        "\n"
        "- **Waterfall** draws only the history from a ready spectrum.\n"
        "- **Spectrum Engine** produces a spectrum for other blocks.\n"
        "- **Slice** picks one head from a multi-head Filter."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_VISUALIZATION_SPECTRUM_ANALYZER_BLOCK_HH
