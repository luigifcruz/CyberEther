#ifndef JETSTREAM_DOMAINS_DSP_PSK_DEMOD_BLOCK_HH
#define JETSTREAM_DOMAINS_DSP_PSK_DEMOD_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct PskDemod : public Block::Config {
    std::string pskType = "qpsk";
    F32 sampleRate = 2000000.0f;
    F32 symbolRate = 1000000.0f;
    F32 frequencyLoopBandwidth = 0.05f;
    F32 timingLoopBandwidth = 0.05f;
    F32 dampingFactor = 0.707f;

    JST_BLOCK_TYPE(psk_demod);
    JST_BLOCK_PARAMS(pskType, sampleRate, symbolRate, frequencyLoopBandwidth,
                     timingLoopBandwidth, dampingFactor);
    JST_BLOCK_DESCRIPTION(
        "PSK Demodulator",
        "Demodulates PSK signals with automatic frequency and timing recovery.",
        "# PSK Demodulator\n"
        "The PSK Demod block demodulates Phase Shift Keying signals (BPSK, QPSK, 8-PSK) from "
        "complex I/Q data streams. It performs carrier frequency recovery using a Costas loop, "
        "symbol timing recovery using a Mueller-Muller detector, and outputs soft symbols. The "
        "block handles fine frequency correction and clock recovery automatically.\n\n"

        "## Arguments\n"
        "- **PSK Type**: Modulation scheme - BPSK (2 symbols), QPSK (4 symbols), or 8-PSK (8 symbols).\n"
        "- **Sample Rate**: Input sample rate in MHz.\n"
        "- **Symbol Rate**: Expected symbol rate in MHz.\n"
        "- **Frequency Loop Bandwidth**: Carrier recovery loop bandwidth (0-1).\n"
        "- **Timing Loop Bandwidth**: Symbol timing recovery loop bandwidth (0-1).\n"
        "- **Damping Factor**: Loop filter damping coefficient for stability.\n\n"

        "## Useful For\n"
        "- Demodulating digital satellite communication signals.\n"
        "- Recovering PSK data from software-defined radio streams.\n"
        "- Digital signal processing in communication systems.\n"
        "- Educational demonstrations of PSK demodulation.\n\n"

        "## Examples\n"
        "- QPSK demodulation:\n"
        "  Config: PSK Type=QPSK, Sample Rate=2MHz, Symbol Rate=0.5MHz\n"
        "  Input: CF32[8192] -> Output: CF32[2048]\n"
        "- BPSK demodulation:\n"
        "  Config: PSK Type=BPSK, Sample Rate=1MHz, Symbol Rate=0.125MHz\n"
        "  Input: CF32[8192] -> Output: CF32[1024]\n\n"

        "## Implementation\n"
        "Input -> Frequency Correction -> Timing Recovery -> Soft Symbol Output\n"
        "1. Costas loop performs carrier phase and frequency tracking.\n"
        "2. Mueller-Muller timing error detector maintains symbol synchronization.\n"
        "3. Linear interpolation provides fractional sample timing.\n"
        "4. Outputs frequency/phase corrected soft symbols without hard mapping.\n"
        "5. Soft symbols preserve amplitude and phase information for further decoding."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_DSP_PSK_DEMOD_BLOCK_HH
