#ifndef JETSTREAM_DOMAINS_IO_SOAPY_BLOCK_HH
#define JETSTREAM_DOMAINS_IO_SOAPY_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct Soapy : public Block::Config {
    std::string hintString = "";
    std::string deviceString = "";
    std::string streamString = "";
    F32 frequency = 96.9e6;
    F32 frequencyStep = 1000000.0;
    F32 sampleRate = 2.0e6;
    bool automaticGain = true;
    U64 numberOfBatches = 8;
    U64 numberOfTimeSamples = 8192;
    U64 bufferMultiplier = 4;

    JST_BLOCK_TYPE(soapy);
    JST_BLOCK_PARAMS(hintString, deviceString, streamString, frequency, frequencyStep,
                     sampleRate, automaticGain, numberOfBatches,
                     numberOfTimeSamples, bufferMultiplier);
    JST_BLOCK_DESCRIPTION(
        "Soapy SDR",
        "Interface for SoapySDR devices.",
        "# SoapySDR Source\n"
        "The Soapy block provides an interface to communicate and control SoapySDR supported devices, "
        "facilitating data acquisition and device configuration.\n\n"

        "## Arguments\n"
        "- **Device Hint**: Filter string for discovering devices.\n"
        "- **Device String**: Full device identifier string.\n"
        "- **Stream String**: Stream configuration arguments.\n"
        "- **Frequency**: Tuner frequency in Hz.\n"
        "- **Sample Rate**: Sampling rate in Hz.\n"
        "- **Automatic Gain**: Enable automatic gain control.\n"
        "- **Number of Batches**: Number of batches in output buffer.\n"
        "- **Number of Time Samples**: Samples per batch.\n"
        "- **Buffer Multiplier**: Internal buffer size multiplier.\n\n"

        "## Useful For\n"
        "- Receiving RF signals from software defined radios.\n"
        "- Real-time spectrum analysis.\n"
        "- Signal processing of live radio data.\n\n"

        "## Examples\n"
        "- Receive FM broadcast:\n"
        "  Config: Frequency=96.9 MHz, Sample Rate=2 MHz, Batches=8, Samples=8192\n"
        "  Output: CF32[8, 8192]\n\n"

        "## Implementation\n"
        "Soapy Module -> Output Buffer\n"
        "1. Opens SoapySDR device with specified configuration.\n"
        "2. Streams samples from device into circular buffer.\n"
        "3. Outputs batches of samples for downstream processing."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_IO_SOAPY_BLOCK_HH
