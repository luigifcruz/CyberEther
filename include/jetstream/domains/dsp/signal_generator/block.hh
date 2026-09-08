#ifndef JETSTREAM_DOMAINS_DSP_SIGNAL_GENERATOR_BLOCK_HH
#define JETSTREAM_DOMAINS_DSP_SIGNAL_GENERATOR_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct SignalGenerator : public Block::Config {
    std::string signalType = "cosine";
    std::string signalDataType = "F32";
    F32 sampleRate = 1000000.0;
    F32 frequency = 1000.0;
    F32 amplitude = 1.0;
    F32 phase = 0.0;
    F32 dcOffset = 0.0;
    F32 noiseVariance = 1.0;
    F32 chirpStartFreq = 1000.0;
    F32 chirpEndFreq = 10000.0;
    F32 chirpDuration = 1.0;
    U64 bufferSize = 8192;

    JST_BLOCK_TYPE(signal_generator);
    JST_BLOCK_DOMAIN("DSP");
    JST_BLOCK_PARAMS(signalType, signalDataType, sampleRate, frequency,
                     amplitude, phase, dcOffset, noiseVariance,
                     chirpStartFreq, chirpEndFreq, chirpDuration, bufferSize);
    JST_BLOCK_DESCRIPTION(
        "Signal Generator",
        "Generates synthetic waveforms, noise, and chirps.",
        "# Signal Generator\n"
        "The Signal Generator block creates zero-centered synthetic baseband signals "
        "for testing and simulation. Controls are shown only when they apply to the "
        "selected waveform.\n\n"

        "## Arguments\n"
        "- **Signal Type**: Sine, Cosine, Square, Triangle, Sawtooth, Noise, "
        "DC, or Chirp.\n"
        "- **Data Type**: F32 produces real samples. CF32 sine, cosine, and "
        "chirp produce analytic IQ; other CF32 waveforms have Q=0.\n"
        "- **Sample Rate**: Sampling frequency in Hz.\n"
        "- **Frequency**: Baseband offset in Hz. Analytic CF32 sinusoids support "
        "signed frequencies; real waveforms use non-negative frequencies.\n"
        "- **Amplitude**: Linear peak multiplier.\n"
        "- **Phase**: Phase offset in radians.\n"
        "- **DC Offset**: Real-valued bias added to samples.\n"
        "- **Noise Variance**: Per-component Gaussian variance before amplitude "
        "scaling.\n"
        "- **Start/End Frequency**: Chirp endpoints in Hz. CF32 chirps may "
        "ascend, descend, or cross zero.\n"
        "- **Duration**: Duration of each repeated chirp sweep.\n"
        "- **Buffer Size**: Samples generated per processing cycle.\n\n"

        "## Useful For\n"
        "- Creating test signals for system verification and debugging.\n"
        "- Generating reference waveforms for signal processing algorithms.\n"
        "- Producing noise sources for statistical analysis.\n"
        "- Creating chirp signals for frequency response measurements.\n\n"

        "## Examples\n"
        "- Complex tone generation:\n"
        "  Config: Signal Type=Cosine, Data Type=CF32, Frequency=-100 kHz, "
        "Sample Rate=2 MHz, Buffer Size=1024\n"
        "  Output: CF32[1024]\n"
        "- Real noise generation:\n"
        "  Config: Signal Type=Noise, Data Type=F32, Noise Variance=0.1, "
        "Buffer Size=8192\n"
        "  Output: F32[8192]\n\n"

        "## Implementation\n"
        "SignalGenerator Module -> Output\n"
        "1. A wrapped F64 phase accumulator maintains long-running precision.\n"
        "2. Frequencies are constrained to the waveform's Nyquist interval.\n"
        "3. Repeated chirps preserve carrier phase at sweep boundaries.\n"
        "4. Frequency, amplitude, phase, and offset updates preserve state.\n"
        "5. Output metadata reports sampleRate and a zero-Hz stream center."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_DSP_SIGNAL_GENERATOR_BLOCK_HH
