#ifndef JETSTREAM_DOMAINS_VISUALIZATION_FRAME_BLOCK_HH
#define JETSTREAM_DOMAINS_VISUALIZATION_FRAME_BLOCK_HH

#include <string>

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct Frame : public Block::Config {
    std::string fit = "contain";
    std::string colormap = "grayscale";
    bool autoRange = true;
    std::string interpolation = "nearest";
    std::string xLabel = "X (px)";
    std::string yLabel = "Y (px)";

    JST_BLOCK_TYPE(frame);
    JST_BLOCK_DOMAIN("Visualization");
    JST_BLOCK_NODE_SIZE(L);
    JST_BLOCK_PARAMS(fit, colormap, autoRange, interpolation, xLabel, yLabel);
    JST_BLOCK_DESCRIPTION(
        "Frame",
        "Displays images and heat maps.",
        "# Frame\n"
        "View images and heat maps, zoom into details, and inspect individual "
        "pixels. Drag to move around, scroll to zoom around the pointer, or "
        "hold the right mouse button and draw a box to zoom into an area. "
        "Right-click to return to the full view.\n\n"

        "## Arguments\n"
        "- **Fit**: Choose how the image fits the viewing area. Contain shows "
        "the whole image, leaving empty space when needed. Cover fills the "
        "area by cropping the edges. Stretch fills the area by changing the "
        "image's proportions.\n"
        "- **Colormap**: Choose grayscale or a color palette for grayscale "
        "images and heat maps.\n"
        "- **Auto Range**: Automatically adjust contrast to bring out the "
        "details in each image. Turn this off to keep a fixed brightness "
        "scale across images.\n"
        "- **Interpolation**: Choose how pixels are sampled when the image is "
        "magnified. Nearest keeps individual pixels sharp. Bilinear smooths "
        "the transitions between pixels. Bicubic fits a smooth curve through "
        "neighboring pixels for the softest result.\n\n"

        "## Input\n"
        "- **Frame**: The image or heat map to display. Supports grayscale, "
        "color, and images with transparency.\n\n"

        "## Useful For\n"
        "- Viewing images from your processing pipeline.\n"
        "- Spotting patterns and differences in heat maps.\n"
        "- Checking pixel positions and values.\n\n"

        "## Implementation\n"
        "Accepts F32 tensors shaped `[height, width]`, `[height, width, 1]`, "
        "`[height, width, 3]`, or `[height, width, 4]`. Values are uploaded "
        "to a GPU storage buffer and rendered on a fitted quad. Auto Range "
        "normalizes color values using each frame's minimum and maximum, "
        "leaving alpha unchanged. With Auto Range off, the display range "
        "is fixed at 0 to 1. Interpolation selects nearest-neighbor sampling, "
        "bilinear filtering, or bicubic (Catmull-Rom) filtering."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_VISUALIZATION_FRAME_BLOCK_HH
