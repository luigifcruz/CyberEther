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
    JST_BLOCK_PARAMS(signalType, signalDataType, sampleRate, frequency, amplitude, phase, dcOffset,
                     noiseVariance, chirpStartFreq, chirpEndFreq, chirpDuration,
                     bufferSize);
    JST_BLOCK_DESCRIPTION(
        "Signal Generator",
        "Generates various types of signals including sine, square, triangle, sawtooth, noise, and chirp.",
        "# Signal Generator\n"
        "The Signal Generator block creates synthetic signals for testing and simulation purposes. "
        "It supports multiple waveform types with configurable parameters for frequency, amplitude, "
        "phase, and other signal-specific properties.\n\n"

        "## Arguments\n"
        "- **Signal Type**: The waveform to generate (Sine, Cosine, Square, "
        "Triangle, Sawtooth, Noise, DC, Chirp).\n"
        "- **Data Type**: The output data type (F32 or CF32).\n"
        "- **Buffer Size**: Number of samples to generate per processing cycle.\n"
        "- **Sample Rate**: The sampling frequency in MHz.\n"
        "- **Frequency**: The fundamental frequency of the signal in MHz.\n"
        "- **Amplitude**: The amplitude scaling factor.\n"
        "- **Phase**: Phase offset in radians.\n"
        "- **DC Offset**: DC bias added to the signal.\n"
        "- **Noise Variance**: Variance of Gaussian noise (for noise type).\n"
        "- **Chirp Start Frequency**: Start frequency for chirp signals in MHz.\n"
        "- **Chirp End Frequency**: End frequency for chirp signals in MHz.\n"
        "- **Chirp Duration**: Duration of one chirp sweep in seconds.\n\n"

        "## Useful For\n"
        "- Creating test signals for system verification and debugging.\n"
        "- Generating reference waveforms for signal processing algorithms.\n"
        "- Producing noise sources for statistical analysis.\n"
        "- Creating chirp signals for frequency response measurements.\n\n"

        "## Examples\n"
        "- Sine wave generation:\n"
        "  Config: Signal Type=Sine, Data Type=CF32, Frequency=0.1 MHz, "
        "Sample Rate=2 MHz, Buffer Size=1024\n"
        "  Output: CF32[1024]\n"
        "- Real noise generation:\n"
        "  Config: Signal Type=Noise, Data Type=F32, Noise Variance=0.1, "
        "Buffer Size=8192\n"
        "  Output: F32[8192]\n\n"

        "## Implementation\n"
        "SignalGenerator Module -> Output\n"
        "1. The signal generator module computes the requested waveform using optimized algorithms.\n"
        "2. For complex outputs (CF32), generates both I and Q components appropriately.\n"
        "3. For real outputs (F32), generates only the real component.\n"
        "4. Maintains phase continuity across buffer boundaries for continuous signals."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_DSP_SIGNAL_GENERATOR_BLOCK_HH
