#ifndef JETSTREAM_DOMAINS_VISUALIZATION_LINEPLOT_BLOCK_HH
#define JETSTREAM_DOMAINS_VISUALIZATION_LINEPLOT_BLOCK_HH

#include <string>
#include <vector>

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct Lineplot : public Block::Config {
    U64 averaging = 1;
    bool maxHold = false;
    bool fill = true;
    F32 rangeMin = -100.0f;
    F32 rangeMax = 0.0f;
    std::string xLabel = "Frequency (MHz)";
    std::string yLabel = "Amplitude (dBFS)";
    std::vector<F32> markers;
    std::vector<U64> pins;

    JST_BLOCK_TYPE(lineplot);
    JST_BLOCK_DOMAIN("Visualization");
    JST_BLOCK_NODE_SIZE(L);
    JST_BLOCK_PARAMS(averaging, maxHold, fill, rangeMin, rangeMax,
                     xLabel, yLabel, markers, pins);
    JST_BLOCK_DESCRIPTION(
        "Lineplot",
        "Displays data in a line plot visualization.",
        "# Lineplot\n"
        "\n"
        "Draws each buffer as a live trace, with optional smoothing and a peak trace. "
        "It usually follows a Spectrum Engine with Enable Scale on, fed by a radio "
        "source.\n"
        "\n"
        "- **Values from 0 to 1 span the plot.** Values past either end are compressed "
        "softly toward the edge.\n"
        "- **The level axis reads -100 to 0 dBFS.** The labels assume that range, "
        "whatever produced the values.\n"
        "- **The axis shows MHz from the input.** It reads `frequency` and `sampleRate`"
        " from the spectrum.\n"
        "- **Controls work in a detached window.** Click the expand icon on the view to"
        " open it.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Input** (in) | F32 | Needs a `sampleAxis` or a `channelAxis` across the "
        "trace with at least 2 values, implied for one-dimensional input. A `batchAxis`"
        " is averaged into one trace, and no other axis is allowed. The buffer must be "
        "contiguous. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **Averaging** | 1 | 1 or more, slider up to 256 | Smooths the trace with an "
        "exponential average over about this many buffers. |\n"
        "| **Max Hold** | Off | On, Off | Adds a trace of the highest level seen, "
        "starting once Averaging buffers have arrived. Changing Averaging clears it. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "All rows assume a 2 MS/s radio feeding a Spectrum Engine with Enable Scale on,"
        " Range Min -100 dBFS, and Range Max 0 dBFS.\n"
        "\n"
        "| Goal | Settings | Input | Output |\n"
        "|---|---|---|---|\n"
        "| Live spectrum | Defaults | 8,192 bins per buffer | A trace of 8,192 points, "
        "updated every 4.1 ms |\n"
        "| Steady trace | Averaging 16 | 8,192 bins per buffer | A trace averaged over "
        "about 66 ms |\n"
        "| Peak search | Max Hold on | 8,192 bins per buffer | The live trace plus the "
        "highest level seen |\n"
        "| Batched spectra | Defaults | 8 batches x 1,024 bins per buffer | A trace of "
        "1,024 points, each the mean of 8 spectra |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The trace hugs the bottom or the top | Input values sit outside 0 to 1, such"
        " as raw dBFS or a waveform from -1 to 1 | Scale the input first, such as with "
        "Enable Scale in the Spectrum Engine. |\n"
        "| Levels on the axis read too high | The Spectrum Engine range is not -100 to "
        "0 dBFS, such as its default Range Min of -120 dBFS | Set Range Min to -100 "
        "dBFS and Range Max to 0 dBFS. |\n"
        "| The trace stops updating | Space froze it, shown by the HOLD label | Press "
        "Space again in its window. |\n"
        "| Zoom and markers do nothing | The view is shown inside the node, which takes"
        " no input | Open it in its own window with the expand icon. |\n"
        "| The block reports an unsupported input data type | The input is not F32, "
        "such as raw complex samples | Feed it a spectrum from a Spectrum Engine. |\n"
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
        "\n"
        "## Under the Hood\n"
        "\n"
        "Batches are averaged, then the trace keeps an exponential average and "
        "compresses values past the range softly instead of clipping. The peak trace "
        "holds the highest displayed value of each point.\n"
        "\n"
        "## See Also\n"
        "\n"
        "- **Spectrum Engine** produces the scaled spectrum it expects.\n"
        "- **Spectrum Analyzer** computes and draws a spectrum in one block.\n"
        "- **Waterfall** draws the same input as a history."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_VISUALIZATION_LINEPLOT_BLOCK_HH
