#ifndef JETSTREAM_DOMAINS_DSP_FM_BLOCK_HH
#define JETSTREAM_DOMAINS_DSP_FM_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct FM : public Block::Config {
    std::string mode = "narrow";
    std::string deemphasis = "none";
    F32 sampleRate = 240e3f;

    JST_BLOCK_TYPE(fm);
    JST_BLOCK_DOMAIN("DSP");
    JST_BLOCK_PARAMS(mode, deemphasis, sampleRate);
    JST_BLOCK_DESCRIPTION(
        "FM Demodulator",
        "Demodulates a frequency modulated signal.",
        "# FM Demodulator\n"
        "The FM block performs quadrature demodulation on complex input signals to recover "
        "the original frequency-modulated baseband content. It uses the instantaneous phase "
        "difference between consecutive samples to extract the modulating signal.\n\n"

        "## Arguments\n"
        "- **Mode**: Narrowband produces mono audio; wideband decodes FM stereo.\n"
        "- **De-emphasis**: Optional de-emphasis time constant (50 or 75 us).\n"
        "- **Sample Rate**: Input signal sample rate in Hz.\n\n"

        "## Useful For\n"
        "- FM broadcast radio demodulation.\n"
        "- WBFM (wideband FM) signal processing.\n"
        "- Analog signal recovery from complex IQ data.\n"
        "- Amateur radio FM reception.\n\n"

        "## Examples\n"
        "- Demodulate narrowband FM at 240 kHz sample rate:\n"
        "  Input: CF32[8192] (IQ samples) -> Output: F32[8192] (mono audio)\n"
        "- Demodulate wideband FM stereo at 240 kHz sample rate:\n"
        "  Input: CF32[8192] (IQ samples) -> Output: F32[8192, 2] (stereo audio)\n\n"

        "## Implementation\n"
        "Input -> FM Demodulator -> Output\n"
        "1. kf = deviation / sampleRate (75 kHz wideband, 100 kHz narrowband)\n"
        "2. ref = 1 / (2 * PI * kf)\n"
        "3. output[n] = arg(conj(input[n-1]) * input[n]) * ref"
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_DSP_FM_BLOCK_HH
