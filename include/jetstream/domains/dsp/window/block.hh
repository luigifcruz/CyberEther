#ifndef JETSTREAM_DOMAINS_DSP_WINDOW_BLOCK_HH
#define JETSTREAM_DOMAINS_DSP_WINDOW_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct Window : public Block::Config {
    U64 size = 1024;

    JST_BLOCK_TYPE(window);
    JST_BLOCK_PARAMS(size);
    JST_BLOCK_DESCRIPTION(
        "Window",
        "Generates a Blackman window function.",
        "# Window\n"
        "The Window block generates a Blackman window of the specified size. "
        "The window coefficients are computed once and cached for efficient reuse. "
        "Blackman windows provide excellent sidelobe suppression, making them ideal "
        "for spectral analysis applications.\n\n"

        "## Arguments\n"
        "- **Size**: Number of samples in the window (window length).\n\n"

        "## Useful For\n"
        "- FFT windowing to reduce spectral leakage.\n"
        "- Filter design applications.\n"
        "- Spectral analysis with low sidelobe levels.\n"
        "- Audio processing and analysis.\n\n"

        "## Examples\n"
        "- Generate a 1024-point Blackman window:\n"
        "  Config: Size=1024\n"
        "  Output: CF32[1024] -> Window coefficients\n\n"

        "## Implementation\n"
        "Window Module -> Output\n"
        "1. Computes Blackman window coefficients: w[n] = 0.42 - 0.5*cos(2*pi*n/(N-1)) + 0.08*cos(4*pi*n/(N-1))\n"
        "2. Coefficients are computed once and cached.\n"
        "3. Output is complex (CF32) with window value as real part and zero imaginary part."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_DSP_WINDOW_BLOCK_HH
