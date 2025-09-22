#ifndef JETSTREAM_DOMAINS_DSP_FFT_BLOCK_HH
#define JETSTREAM_DOMAINS_DSP_FFT_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct Fft : public Block::Config {
    bool forward = true;

    JST_BLOCK_TYPE(fft);
    JST_BLOCK_PARAMS(forward);
    JST_BLOCK_DESCRIPTION(
        "FFT",
        "Performs the Fast Fourier Transform.",
        "# FFT\n"
        "The FFT block computes the Fast Fourier Transform of the input signal, converting "
        "time-domain data to frequency-domain representation. It supports both forward and "
        "inverse transforms for complex and real data types.\n\n"

        "## Arguments\n"
        "- **Direction**: Forward FFT converts time-domain to frequency-domain. "
        "Inverse FFT converts frequency-domain back to time-domain.\n\n"

        "## Useful For\n"
        "- Spectral analysis of signals.\n"
        "- Frequency filtering and manipulation.\n"
        "- Converting between time and frequency domains.\n"
        "- Power spectrum computation.\n\n"

        "## Examples\n"
        "- Complex FFT:\n"
        "  Input: CF32[1024] -> Output: CF32[1024]\n"
        "- Real-to-complex FFT:\n"
        "  Input: F32[1024] -> Output: CF32[1024]\n\n"

        "## Implementation\n"
        "Input Buffer -> FFT Module -> Output Buffer\n"
        "1. Input signal is passed to the FFT computation kernel.\n"
        "2. PocketFFT library performs the transform on CPU.\n"
        "3. Output contains the frequency-domain representation."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_DSP_FFT_BLOCK_HH
