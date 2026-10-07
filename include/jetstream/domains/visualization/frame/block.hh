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
        "\n"
        "Draws an F32 tensor of rows and columns as a grayscale, color-mapped, RGB, or "
        "RGBA image. It usually follows a Reshape or a Python block that arranges "
        "values into rows and columns.\n"
        "\n"
        "- **Auto Range rescales every frame.** Brightness follows the lowest and "
        "highest value of each frame, so frames are not comparable.\n"
        "- **Controls work in a detached window.** Click the expand icon on the view to"
        " open it.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Frame** (in) | F32 | Rows x Columns, or Rows x Columns x Channels with 1, "
        "3, or 4 channels for gray, RGB, or RGBA. Row 0 is at the top. The buffer must "
        "be contiguous. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **Fit** | Contain | Contain, Cover, Stretch | Contain shows the whole frame."
        " Cover fills the view and crops the edges. Stretch fills the view and changes "
        "the proportions. |\n"
        "| **Colormap** | Grayscale | Grayscale, Turbo, Viridis, Inferno, Magma, "
        "Plasma, Jet | Palette from low to high values. Shown only for single-channel "
        "frames. |\n"
        "| **Auto Range** | On | On, Off | Maps the lowest and highest color value of "
        "each frame to the full display range. Off shows values from 0 to 1, and alpha "
        "is never ranged. |\n"
        "| **Interpolation** | Nearest | Nearest, Bilinear, Bicubic | How values "
        "between pixel centers are drawn. Nearest keeps pixels sharp, and Bicubic is "
        "the smoothest. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "| Goal | Settings | Input | Output |\n"
        "|---|---|---|---|\n"
        "| Heat map | Colormap Turbo | 256 rows x 512 columns | A 512 by 256 pixel "
        "image in Turbo colors |\n"
        "| Photo in its own colors | Auto Range off | 480 rows x 640 columns x 3 "
        "channels, from 0 to 1 | The image as stored, with no contrast stretch |\n"
        "| Fill a square view | Fit Cover | 1,080 rows x 1,920 columns x 3 channels | "
        "About the middle 1,080 columns, with the sides cropped |\n"
        "| Smooth close-up | Interpolation Bicubic | 64 rows x 64 columns, zoomed 32x |"
        " Smooth gradients instead of square blocks |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The image is black or washed out | Auto Range is off and the values sit "
        "outside 0 to 1 | Turn on Auto Range, or scale the input to 0 to 1. |\n"
        "| The block reports an invalid input rank, expected 2 or 3 | The input is "
        "one-dimensional, such as a spectrum, or has four axes or more | Arrange it "
        "into rows and columns with a Reshape block. |\n"
        "| The block reports an invalid channel count | The last axis of a "
        "three-dimensional input is not 1, 3, or 4 long | Feed a two-dimensional frame,"
        " or Slice the last axis down to 1, 3, or 4. |\n"
        "| The block reports an unsupported input data type | The input is not F32 | "
        "Convert it with a Cast block. |\n"
        "| The block reports that it expects a contiguous tensor | The input is "
        "strided, such as a Slice with Contiguous off | Turn on Contiguous in the "
        "Slice. |\n"
        "\n"
        "## Controls\n"
        "\n"
        "| Action | Input |\n"
        "|---|---|\n"
        "| Zoom around the cursor, up to 64x | Scroll |\n"
        "| Pan while zoomed | Drag |\n"
        "| Zoom into a box | Right Drag |\n"
        "| Reset zoom and pan | Right Click |\n"
        "| Show the position and value of a pixel | Hover |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "Each frame is uploaded whole and drawn by a shader that samples it with the "
        "chosen interpolation, where Bicubic uses a Catmull-Rom curve. Auto Range takes"
        " the lowest and highest finite value across the color channels of the latest "
        "frame, skipping alpha. Single-channel values go through the colormap after "
        "ranging.\n"
        "\n"
        "## See Also\n"
        "\n"
        "- **Reshape** arranges a stream into rows for this view.\n"
        "- **Spectrogram** shows how often each level occurs per bin.\n"
        "- **Waterfall** shows spectra as a scrolling image.\n"
        "- **Cast** converts other types to F32."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_VISUALIZATION_FRAME_BLOCK_HH
