#ifndef JETSTREAM_DOMAINS_VISUALIZATION_WATERFALL_BLOCK_HH
#define JETSTREAM_DOMAINS_VISUALIZATION_WATERFALL_BLOCK_HH

#include <string>
#include <vector>

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct Waterfall : public Block::Config {
    U64 height = 1024;
    U64 averaging = 1;
    std::string colormap = "turbo";
    std::string xLabel = "Frequency (MHz)";
    std::string yLabel = "Time";
    std::vector<F32> markers;
    std::vector<U64> pins;

    JST_BLOCK_TYPE(waterfall);
    JST_BLOCK_DOMAIN("Visualization");
    JST_BLOCK_NODE_SIZE(L);
    JST_BLOCK_PARAMS(height, averaging, colormap, xLabel, yLabel, markers, pins);
    JST_BLOCK_DESCRIPTION(
        "Waterfall",
        "Shows frequency spectrum over time as a scrolling waterfall.",
        "# Waterfall\n"
        "\n"
        "Draws a scrolling color history of spectra, adding one row per buffer with the"
        " newest at the top. It usually follows a Spectrum Engine with Enable Scale on,"
        " fed by a radio source.\n"
        "\n"
        "- **Values from 0 to 1 span the colormap.** Raw dBFS values fall below that "
        "range and show one flat color.\n"
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
        "row, implied for one-dimensional input. A `batchAxis` adds one row per batch, "
        "and no other axis is allowed. The buffer must be contiguous. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **Height** | 1,024 rows | 1 to 8,192 | Rows of history on screen. |\n"
        "| **Averaging** | 1 | 1 or more, slider up to 64 | Consecutive rows averaged "
        "into each displayed row. |\n"
        "| **Colormap** | Turbo | Grayscale, Turbo, Viridis, Inferno, Magma, Plasma, "
        "Jet | Palette from low to high values. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "All rows assume a 2 MS/s radio feeding a Spectrum Engine with Enable Scale on.\n"
        "\n"
        "| Goal | Settings | Input | Output |\n"
        "|---|---|---|---|\n"
        "| Live history | Defaults | 8,192 bins per buffer | 1,024 rows covering 4.2 "
        "seconds |\n"
        "| Longer, smoother history | Averaging 8 | 8,192 bins per buffer | 1,024 rows "
        "covering 33.6 seconds |\n"
        "| Finer time steps | Height 2,048 | 8 batches x 1,024 bins per buffer | 2,048 "
        "rows covering 1.05 seconds |\n"
        "| Unscaled spectrum | Defaults, with Enable Scale off | 8,192 bins in dBFS | "
        "Every row in the lowest color |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The waterfall is one flat color | Input values sit outside 0 to 1, such as "
        "raw dBFS | Turn on Enable Scale in the Spectrum Engine and set its range "
        "around the signal. |\n"
        "| The view stops scrolling | Space froze it, and the Waterfall shows no HOLD "
        "label | Press Space again in its window. |\n"
        "| Zoom and markers do nothing | The view is shown inside the node, which takes"
        " no input | Open it in its own window with the expand icon. |\n"
        "| The block reports an unsupported input data type | The input is not F32, "
        "such as raw complex samples | Feed it a spectrum from a Spectrum Engine. |\n"
        "| The block reports that the input cannot contain both `sampleAxis` and "
        "`channelAxis` | The spectrum has a `channelAxis`, such as one from a "
        "multi-head Filter | Pick one head with a Slice block. |\n"
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
        "Rows go into a ring buffer of Height rows, and each averaged row is the plain "
        "mean of consecutive spectra. The display blurs a few neighboring rows in time "
        "and passes each value through a soft contrast curve before the colormap "
        "lookup.\n"
        "\n"
        "## See Also\n"
        "\n"
        "- **Spectrum Engine** produces the scaled spectrum it expects.\n"
        "- **Spectrum Analyzer** pairs a live trace with a waterfall.\n"
        "- The same history as a surface is the **3D Waterfall**.\n"
        "- **Slice** picks one head from a multi-head spectrum."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_VISUALIZATION_WATERFALL_BLOCK_HH
