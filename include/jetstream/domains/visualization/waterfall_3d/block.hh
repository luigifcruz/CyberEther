#ifndef JETSTREAM_DOMAINS_VISUALIZATION_WATERFALL_3D_BLOCK_HH
#define JETSTREAM_DOMAINS_VISUALIZATION_WATERFALL_3D_BLOCK_HH

#include <string>

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct Waterfall3D : public Block::Config {
    U64 height = 256;
    U64 averaging = 1;
    std::string colormap = "turbo";
    std::string xLabel = "Frequency (MHz)";
    std::string timeLabel = "Time (rows)";
    std::string amplitudeLabel = "Amplitude";

    JST_BLOCK_TYPE(waterfall_3d);
    JST_BLOCK_DOMAIN("Visualization");
    JST_BLOCK_NODE_SIZE(XL);
    JST_BLOCK_PARAMS(height, averaging, colormap, xLabel, timeLabel, amplitudeLabel);
    JST_BLOCK_DESCRIPTION(
        "3D Waterfall",
        "Displays spectrum history as a 3D surface.",
        "# 3D Waterfall\n"
        "\n"
        "Draws a scrolling history of spectra as a shaded surface, adding one row per "
        "buffer with the newest nearest the camera. It usually follows a Spectrum "
        "Engine with Enable Scale on, fed by a radio source.\n"
        "\n"
        "- **Values from 0 to 1 span the surface.** Raw dBFS values lie flat on the "
        "floor in the lowest color.\n"
        "- **The axis shows MHz from the input.** It reads `frequency` and `sampleRate`"
        " from the spectrum.\n"
        "- **The surface has at most 768 columns.** Each keeps the peak of the bins it "
        "covers, so narrow signals stay visible.\n"
        "- **Controls work in a detached window.** Click the expand icon on the view to"
        " open it.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Input** (in) | F32 | Needs a `sampleAxis` or a `channelAxis` across the "
        "row with at least 2 values, implied for one-dimensional input. A `batchAxis` "
        "adds one row per batch, and no other axis is allowed. The buffer must be "
        "contiguous. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **Height** | 256 rows | 2 to 8,192 | Rows of history on the surface. |\n"
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
        "| Live surface | Defaults | 8,192 bins per buffer | 256 rows covering 1.05 "
        "seconds |\n"
        "| Longer history | Height 1,024 | 8,192 bins per buffer | 1,024 rows covering "
        "4.2 seconds |\n"
        "| Smoother surface | Averaging 8 | 8,192 bins per buffer | 256 rows covering "
        "8.4 seconds |\n"
        "| Finer time steps | Defaults | 8 batches x 1,024 bins per buffer | 256 rows "
        "covering 0.13 seconds |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The surface is flat and one color | Input values sit outside 0 to 1, such as"
        " raw dBFS | Turn on Enable Scale in the Spectrum Engine and set its range "
        "around the signal. |\n"
        "| The surface stops scrolling | Space froze it, shown by the HOLD label | "
        "Press Space again in its window. |\n"
        "| Orbit and zoom do nothing | The view is shown inside the node, which takes "
        "no input | Open it in its own window with the expand icon. |\n"
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
        "| Freeze or resume the view | Space |\n"
        "| Orbit around the surface | Drag |\n"
        "| Pan the view | Right Drag |\n"
        "| Move closer or farther | Scroll |\n"
        "| Return to the home view | Right Click |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "Rows go into a ring buffer of Height rows, and each averaged row is the plain "
        "mean of consecutive spectra. The display blurs a few neighboring rows in time,"
        " and a soft contrast curve sets the height and the color of each point.\n"
        "\n"
        "## See Also\n"
        "\n"
        "- **Waterfall** draws the same history as a flat image.\n"
        "- **Spectrum Engine** produces the scaled spectrum it expects.\n"
        "- **Spectrum Analyzer** has a 3D Waterfall mode with its own spectrum.\n"
        "- **Slice** picks one head from a multi-head spectrum."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_VISUALIZATION_WATERFALL_3D_BLOCK_HH
