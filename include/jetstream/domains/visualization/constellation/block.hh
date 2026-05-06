#ifndef JETSTREAM_DOMAINS_VISUALIZATION_CONSTELLATION_BLOCK_HH
#define JETSTREAM_DOMAINS_VISUALIZATION_CONSTELLATION_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct Constellation : public Block::Config {
    JST_BLOCK_TYPE(constellation);
    JST_BLOCK_DOMAIN("Visualization");
    JST_BLOCK_PARAMS();
    JST_BLOCK_DESCRIPTION(
        "Constellation",
        "Displays a constellation scatter plot.",
        "# Constellation\n"
        "The Constellation block visualizes complex-valued data as a 2D scatter "
        "plot, mapping the real component to the X axis and the imaginary component "
        "to the Y axis. Commonly used in digital communications to display symbol "
        "modulation.\n\n"

        "## Useful For\n"
        "- Visualizing QAM, PSK, and other modulation schemes.\n"
        "- Assessing signal quality and noise characteristics.\n"
        "- Monitoring demodulated symbol positions.\n\n"

        "## Examples\n"
        "- QPSK constellation:\n"
        "  Input: CF32[2048] -> 2D scatter of demodulated symbols.\n\n"

        "## Implementation\n"
        "Input -> Complex to 2D Mapping -> Shape Rendering -> Display\n"
        "1. Complex input values are decomposed into real (X) and imaginary (Y).\n"
        "2. Each sample is rendered as a circle on the scatter plot.\n"
        "3. The plot is rendered to a framebuffer surface.";
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_VISUALIZATION_CONSTELLATION_BLOCK_HH
