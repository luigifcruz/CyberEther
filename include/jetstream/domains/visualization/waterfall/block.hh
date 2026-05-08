#ifndef JETSTREAM_DOMAINS_VISUALIZATION_WATERFALL_BLOCK_HH
#define JETSTREAM_DOMAINS_VISUALIZATION_WATERFALL_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct Waterfall : public Block::Config {
    U64 height = 512;
    bool interpolate = true;

    JST_BLOCK_TYPE(waterfall);
    JST_BLOCK_DOMAIN("Visualization");
    JST_BLOCK_PARAMS(height, interpolate);
    JST_BLOCK_DESCRIPTION(
        "Waterfall",
        "Shows frequency spectrum over time as a scrolling waterfall.",
        "# Waterfall\n"
        "The Waterfall block provides a time-frequency visualization where signal "
        "amplitude is represented using colors. New data appears at the top and "
        "scrolls downward, creating a visual history of the signal's spectral "
        "content over time.\n\n"

        "## Arguments\n"
        "- **Height**: Number of rows in the waterfall history buffer.\n"
        "- **Interpolate**: Enable texture interpolation for smoother visual "
        "appearance.\n\n"

        "## Useful For\n"
        "- Visualizing RF spectrum over time in SDR applications.\n"
        "- Identifying periodic signals and interference patterns.\n"
        "- Analyzing time-varying frequency content in audio or radio signals.\n\n"

        "## Examples\n"
        "- Waterfall display of FFT output:\n"
        "  Config: Height=512, Interpolate=true\n"
        "  Input: F32[1024] -> Scrolling color-mapped display.\n\n"

        "## Implementation\n"
        "Input -> Frequency Bins Buffer -> GPU Texture -> Rendered Display\n"
        "1. Input data is written to a circular buffer of frequency bins.\n"
        "2. The buffer is uploaded to a GPU texture for rendering.\n"
        "3. A color lookup table maps amplitude values to colors.\n"
        "4. The texture is rendered with scrolling to show temporal history.";
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_VISUALIZATION_WATERFALL_BLOCK_HH
