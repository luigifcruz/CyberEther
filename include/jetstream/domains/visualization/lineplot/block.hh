#ifndef JETSTREAM_DOMAINS_VISUALIZATION_LINEPLOT_BLOCK_HH
#define JETSTREAM_DOMAINS_VISUALIZATION_LINEPLOT_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct Lineplot : public Block::Config {
    U64 averaging = 1;
    U64 decimation = 1;
    U64 numberOfVerticalLines = 11;
    U64 numberOfHorizontalLines = 5;
    F32 thickness = 1.0f;

    JST_BLOCK_TYPE(lineplot);
    JST_BLOCK_PARAMS(averaging, decimation, numberOfVerticalLines, numberOfHorizontalLines, thickness);
    JST_BLOCK_DESCRIPTION(
        "Lineplot",
        "Displays data in a line plot visualization.",
        "# Line Plot\n"
        "The Lineplot block visualizes input data as a line graph, suitable "
        "for time-domain signals, waveform displays, and spectral data.\n\n"

        "## Arguments\n"
        "- **Averaging**: Number of samples to average for smoothing "
        "(1 = no averaging).\n"
        "- **Decimation**: Decimation factor for input data (1 = no decimation).\n"
        "- **Number of Vertical Lines**: Number of vertical grid lines.\n"
        "- **Number of Horizontal Lines**: Number of horizontal grid lines.\n"
        "- **Thickness**: Line thickness multiplier.\n\n"

        "## Useful For\n"
        "- Visualizing time-domain signals and waveforms.\n"
        "- Displaying FFT magnitude spectra.\n"
        "- Real-time signal monitoring.\n\n"

        "## Examples\n"
        "- Spectrum display with averaging:\n"
        "  Config: Averaging=8, Decimation=1\n"
        "  Input: F32[1024] -> Rendered line plot.\n\n"

        "## Implementation\n"
        "Input -> Signal Points -> GPU Vertices -> Rendered Display\n"
        "1. Input data is processed with averaging and decimation.\n"
        "2. Signal points are computed for each sample.\n"
        "3. Thick line vertices are generated on GPU for rendering.\n"
        "4. Grid, signal, and cursor are rendered to a framebuffer.";
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_VISUALIZATION_LINEPLOT_BLOCK_HH
