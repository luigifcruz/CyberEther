#ifndef JETSTREAM_DOMAINS_VISUALIZATION_SPECTROGRAM_BLOCK_HH
#define JETSTREAM_DOMAINS_VISUALIZATION_SPECTROGRAM_BLOCK_HH

#include <string>

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct Spectrogram : public Block::Config {
    U64 height = 256;
    std::string colormap = "turbo";
    std::string xLabel = "Frequency (MHz)";
    std::string yLabel = "Magnitude";

    JST_BLOCK_TYPE(spectrogram);
    JST_BLOCK_DOMAIN("Visualization");
    JST_BLOCK_NODE_SIZE(L);
    JST_BLOCK_PARAMS(height, colormap, xLabel, yLabel);
    JST_BLOCK_DESCRIPTION(
        "Spectrogram",
        "Displays a spectrogram of data.",
        "# Spectrogram\n"
        "\n"
        "Draws how often each level occurs in every bin, as a color density map that "
        "fades over time. It usually follows a Spectrum Engine with Enable Scale on, "
        "fed by a radio source.\n"
        "\n"
        "- **Colors count hits, not power.** Each value adds a hit at its level, and a "
        "level saturates after about 50 hits.\n"
        "- **Values from 0 to 1 span the height.** Values in the lowest level, or "
        "outside that range, are not drawn.\n"
        "- **Old hits fade with every batch.** More batches per second clear the plot "
        "faster.\n"
        "- **Controls work in a detached window.** Click the expand icon on the view to"
        " open it.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Input** (in) | F32 | Needs a `sampleAxis` or a `channelAxis` across the "
        "plot, implied for one-dimensional input. A `batchAxis` adds one hit per batch,"
        " and no other axis is allowed. The buffer must be contiguous. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **Height** | 256 levels | 1 to 2,048 | Levels on the vertical axis, from "
        "input value 0 at the bottom to 1 at the top. |\n"
        "| **Colormap** | Turbo | Grayscale, Turbo, Viridis, Inferno, Magma, Plasma, "
        "Jet | Palette from rare to frequent levels. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "All rows assume a 2 MS/s radio feeding a Spectrum Engine with Enable Scale on,"
        " Range Min -100 dBFS, and Range Max 0 dBFS.\n"
        "\n"
        "| Goal | Settings | Input | Output |\n"
        "|---|---|---|---|\n"
        "| Signal density | Defaults | 8,192 bins per buffer | 256 levels of 0.39 dB, "
        "with a lone hit fading to half in 2.8 seconds |\n"
        "| Finer levels | Height 1,024 | 8,192 bins per buffer | 1,024 levels of 0.1 dB"
        " |\n"
        "| Coarser levels | Height 64 | 8,192 bins per buffer | 64 levels of 1.56 dB |\n"
        "| Faster fade | Defaults | 8 batches x 1,024 bins per buffer | 1,024 columns, "
        "with a lone hit fading to half in 0.35 seconds |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The plot stays empty | Input values sit outside 0 to 1, such as raw dBFS | "
        "Turn on Enable Scale in the Spectrum Engine and set its range around the "
        "signal. |\n"
        "| The noise floor is missing | It sits at or below the lowest level, which is "
        "never drawn | Lower Range Min in the Spectrum Engine. |\n"
        "| Zoom does nothing | The view is shown inside the node, which takes no input "
        "| Open it in its own window with the expand icon. |\n"
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
        "\n"
        "## Under the Hood\n"
        "\n"
        "Each buffer fades every level slightly, then adds a fixed amount to the level "
        "each value lands on, up to full color. The display smooths the levels with "
        "bicubic interpolation before the colormap lookup.\n"
        "\n"
        "## See Also\n"
        "\n"
        "- **Spectrum Engine** produces the scaled spectrum it expects.\n"
        "- **Waterfall** shows the same spectra as a history over time.\n"
        "- **Lineplot** shows only the latest spectrum as a trace."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_VISUALIZATION_SPECTROGRAM_BLOCK_HH
