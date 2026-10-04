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
        "The 3D Waterfall block lifts the scrolling waterfall into a surface. "
        "Frequency runs across the grid, time runs into the distance with the "
        "newest row closest to the camera, and amplitude raises each point. "
        "Colors follow the same map as the flat waterfall and blend smoothly "
        "between neighboring bins. A bathtub frame surrounds the surface with "
        "the two far walls carrying the grid and the two near walls left open. "
        "The sides of the surface facing the camera drop down to the floor, "
        "colored by height, so the surface reads as a solid block. "
        "A trace along the newest row shows where new spectra enter.\n\n"

        "## Arguments\n"
        "- **Height**: Number of rows in the history buffer.\n"
        "- **Averaging**: Number of spectra averaged per displayed row.\n"
        "- **Colormap**: Color palette that maps amplitude to color.\n\n"

        "## Interaction\n"
        "- Drag with the left button to orbit around the surface.\n"
        "- Drag with the right button to pan the view.\n"
        "- Scroll to move closer or farther away.\n"
        "- Click the right button without dragging to return home.\n"
        "- Press **Space** to freeze or resume the surface.\n\n"

        "## Useful For\n"
        "- Reading weak signals whose shape is lost in a flat color map.\n"
        "- Comparing the amplitude of bursts across time at a glance.\n"
        "- Presenting spectrum history in demonstrations and recordings.\n\n"

        "## Examples\n"
        "- Surface view of FFT output:\n"
        "  Config: Height=256\n"
        "  Input: F32[1024] -> Shaded surface with an orbit camera.\n\n"

        "## Implementation\n"
        "Input -> Signal View (waterfall_3d mode) -> Rendered Display\n"
        "1. Input data is written to a circular buffer of frequency bins.\n"
        "2. Each row is reduced to one column per few pixels of view width "
        "by keeping the peak of the nearby bins, then uploaded to a GPU "
        "storage buffer.\n"
        "3. A vertex shader builds one quad per column and raises it with the "
        "same curve as the colors, then shades it from the wider "
        "neighborhood.\n"
        "4. Cells are emitted far to near so painting order handles occlusion.";
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_VISUALIZATION_WATERFALL_3D_BLOCK_HH
