#ifndef JETSTREAM_DOMAINS_DSP_SPECTRUM_ENGINE_BLOCK_HH
#define JETSTREAM_DOMAINS_DSP_SPECTRUM_ENGINE_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct SpectrumEngine : public Block::Config {
    U64 axis = 1;
    bool enableAgc = false;
    bool enableScale = false;
    F32 rangeMin = -120.0f;
    F32 rangeMax = 0.0f;

    JST_BLOCK_TYPE(spectrum_engine);
    JST_BLOCK_DOMAIN("DSP");
    JST_BLOCK_PARAMS(axis, enableAgc, enableScale, rangeMin, rangeMax);
    JST_BLOCK_DESCRIPTION(
        "Spectrum Engine",
        "Computes spectra with windowing, FFT, and optional scaling.",
        "# Spectrum Engine\n"
        "The Spectrum Engine block computes the frequency spectrum of the input "
        "signal through a configurable processing chain. It applies a Blackman "
        "window, performs FFT, and optionally applies AGC and range scaling to "
        "produce the final spectrum output in decibels.\n\n"

        "## Arguments\n"
        "- **Axis**: The axis along which to compute the spectrum (determines "
        "window size).\n"
        "- **Enable AGC**: Whether to apply automatic gain control after the FFT.\n"
        "- **Enable Scale**: Whether to apply range scaling to the output.\n"
        "- **Range Min**: Minimum value of the scale range (dBFS). Only used "
        "when Enable Scale is true.\n"
        "- **Range Max**: Maximum value of the scale range (dBFS). Only used "
        "when Enable Scale is true.\n\n"

        "## Useful For\n"
        "- Spectral analysis and visualization.\n"
        "- Power spectral density computation.\n"
        "- Frequency domain signal monitoring.\n\n"

        "## Examples\n"
        "- Batched spectrum analysis:\n"
        "  Config: Axis=1, Enable AGC=false, Enable Scale=true, Range Min=-120, "
        "Range Max=0\n"
        "  Input: CF32[8, 1024] -> Output: F32[8, 1024]\n\n"

        "## Implementation\n"
        "Input -> Window + Invert -> Multiply -> FFT -> [AGC] -> Amplitude -> "
        "[Range] -> Output\n"
        "1. Window module generates a Blackman window sized to the input axis.\n"
        "2. Invert module applies FFT shift to center the window.\n"
        "3. Multiply module applies the shifted window to the input signal.\n"
        "4. FFT module computes the forward Fourier transform.\n"
        "5. Optional AGC module normalizes signal amplitude.\n"
        "6. Amplitude module computes the magnitude in decibels.\n"
        "7. Optional Range module scales the output to the specified range.";
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_DSP_SPECTRUM_ENGINE_BLOCK_HH
