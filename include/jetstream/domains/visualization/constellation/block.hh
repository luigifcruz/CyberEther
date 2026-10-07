#ifndef JETSTREAM_DOMAINS_VISUALIZATION_CONSTELLATION_BLOCK_HH
#define JETSTREAM_DOMAINS_VISUALIZATION_CONSTELLATION_BLOCK_HH

#include <string>

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct Constellation : public Block::Config {
    std::string xLabel = "In-Phase";
    std::string yLabel = "Quadrature";

    JST_BLOCK_TYPE(constellation);
    JST_BLOCK_DOMAIN("Visualization");
    JST_BLOCK_NODE_SIZE(L);
    JST_BLOCK_PARAMS(xLabel, yLabel);
    JST_BLOCK_DESCRIPTION(
        "Constellation",
        "Displays a constellation scatter plot.",
        "# Constellation\n"
        "\n"
        "Plots every complex value of each buffer as a dot, with the real part across "
        "and the imaginary part up. It usually follows a PSK Demodulator, to show "
        "whether its symbols form tight clusters.\n"
        "\n"
        "- **Both axes span -1 to 1.** Values beyond that are cut off until you zoom "
        "out.\n"
        "- **Each buffer replaces the last.** Only the newest buffer is drawn, with no "
        "persistence.\n"
        "- **Controls work in a detached window.** Click the expand icon on the view to"
        " open it.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Input** (in) | CF32 | One or two dimensions, with every value drawn as one"
        " dot. The buffer must be contiguous. |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| Dots crowd the center or vanish | Values sit far below or beyond 1, such as "
        "raw radio samples | Put an AGC upstream, or zoom with Scroll in a detached "
        "window. |\n"
        "| The block reports an unsupported input data type | The input is not CF32, "
        "such as real audio | Feed it complex samples or symbols. |\n"
        "| The block reports an invalid input rank, expected 1 or 2 | The input has "
        "three or more dimensions, such as batches of heads from a Filter | Pick one "
        "head with a Slice block. |\n"
        "| The block reports that a contiguous tensor is expected | The input is "
        "strided, such as a Slice with Contiguous off | Turn on Contiguous in the "
        "Slice. |\n"
        "\n"
        "## Controls\n"
        "\n"
        "| Action | Input |\n"
        "|---|---|\n"
        "| Zoom both axes around the center, with no limit | Scroll |\n"
        "| Reset zoom | Right Click |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "Each value becomes one circle at its real and imaginary position, and circles "
        "outside the plot frame are clipped. Zoom scales the positions and the tick "
        "labels together.\n"
        "\n"
        "## See Also\n"
        "\n"
        "- Symbols to check come from the **PSK Demodulator**.\n"
        "- Levels near 1 come from an **AGC** upstream.\n"
        "- **Slice** picks one head from a multi-head Filter.\n"
        "- **Spectrum Analyzer** shows the same samples as a spectrum."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_VISUALIZATION_CONSTELLATION_BLOCK_HH
