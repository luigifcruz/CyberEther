#ifndef JETSTREAM_DOMAINS_VISUALIZATION_SPECTROGRAM_BLOCK_HH
#define JETSTREAM_DOMAINS_VISUALIZATION_SPECTROGRAM_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct Spectrogram : public Block::Config {
    U64 height = 256;

    JST_BLOCK_TYPE(spectrogram);
    JST_BLOCK_PARAMS(height);
    JST_BLOCK_DESCRIPTION(
        "Spectrogram",
        "Displays a spectrogram of data.",
        "# Spectrogram\n"
        "The Spectrogram block visualizes how frequencies of input data change "
        "over time. It represents amplitude of frequencies using color intensity "
        "in a 2D histogram that accumulates and decays.\n\n"

        "## Arguments\n"
        "- **Height**: Number of frequency bins in the vertical axis.\n\n"

        "## Useful For\n"
        "- Visualizing frequency distribution as a heat map.\n"
        "- Identifying spectral peaks and harmonic structures.\n"
        "- Observing the density of spectral activity over time.\n\n"

        "## Examples\n"
        "- Spectrogram display of FFT output:\n"
        "  Config: Height=256\n"
        "  Input: F32[1024] -> Accumulated frequency heat map.\n\n"

        "## Implementation\n"
        "Input -> Frequency Bins Accumulation -> GPU Texture -> Rendered Display\n"
        "1. Input values are mapped to vertical bins and accumulated.\n"
        "2. The accumulated bins decay over time to show temporal density.\n"
        "3. A color lookup table maps intensity values to colors.\n"
        "4. Bicubic interpolation provides smooth rendering.";
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_VISUALIZATION_SPECTROGRAM_BLOCK_HH
