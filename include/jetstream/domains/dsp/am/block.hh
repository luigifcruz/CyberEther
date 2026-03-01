#ifndef JETSTREAM_DOMAINS_DSP_AM_BLOCK_HH
#define JETSTREAM_DOMAINS_DSP_AM_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct AM : public Block::Config {
    F32 sampleRate = 240e3f;
    F32 dcAlpha = 0.995f;

    JST_BLOCK_TYPE(am);
    JST_BLOCK_PARAMS(sampleRate, dcAlpha);
    JST_BLOCK_DESCRIPTION(
        "AM Demodulator",
        "Demodulates an amplitude modulated signal.",
        "# AM Demodulator\n"
        "The AM block performs envelope detection on complex input signals to recover "
        "the original amplitude-modulated baseband content. It computes the magnitude "
        "of each complex sample and applies a DC-blocking filter to remove the carrier "
        "component.\n\n"

        "## Arguments\n"
        "- **Sample Rate**: Input signal sample rate in Hz.\n"
        "- **DC Alpha**: DC-blocking filter coefficient (0 to 1). Higher values "
        "remove less low-frequency content.\n\n"

        "## Useful For\n"
        "- AM broadcast radio demodulation.\n"
        "- Shortwave AM signal processing.\n"
        "- Analog signal recovery from complex IQ data.\n"
        "- Aviation radio AM reception.\n\n"

        "## Examples\n"
        "- Demodulate AM broadcast at 240 kHz sample rate:\n"
        "  Input: CF32[8192] (IQ samples) -> Output: F32[8192] (audio)\n\n"

        "## Implementation\n"
        "Input -> Envelope Detector -> DC Blocker -> Output\n"
        "1. envelope[n] = |input[n]| (magnitude of complex sample)\n"
        "2. output[n] = envelope[n] - envelope[n-1] + alpha * output[n-1]"
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_DSP_AM_BLOCK_HH
