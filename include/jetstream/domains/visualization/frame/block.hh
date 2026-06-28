#ifndef JETSTREAM_DOMAINS_VISUALIZATION_FRAME_BLOCK_HH
#define JETSTREAM_DOMAINS_VISUALIZATION_FRAME_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct Frame : public Block::Config {
    bool lut = false;

    JST_BLOCK_TYPE(frame);
    JST_BLOCK_DOMAIN("Visualization");
    JST_BLOCK_PARAMS(lut);
    JST_BLOCK_DESCRIPTION(
        "Frame",
        "Displays a frame buffer on a surface.",
        "# Frame\n"
        "The Frame block renders a 2D F32 frame tensor directly to a surface. "
        "Scalar frames are shown as grayscale by default, with an optional Turbo "
        "lookup table for color mapping. RGB and RGBA frames are rendered directly.\n\n"

        "## Arguments\n"
        "- **LUT**: Apply the Turbo color lookup table to scalar values. "
        "Disabled by default.\n\n"

        "## Input\n"
        "- **Frame**: F32 tensor shaped `[height, width]`, `[height, width, 3]`, "
        "or `[height, width, 4]`.\n\n"

        "## Useful For\n"
        "- Displaying image frames produced by a processing pipeline.\n"
        "- Visualizing scalar heat maps with an optional LUT.\n"
        "- Rendering RGB/RGBA frame buffers without axes or plot overlays.\n\n"

        "## Implementation\n"
        "Input Buffer -> GPU Storage Buffer -> Fullscreen Quad -> Rendered Surface"
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_VISUALIZATION_FRAME_BLOCK_HH
