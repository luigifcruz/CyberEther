#ifndef JETSTREAM_DOMAINS_IO_AUDIO_BLOCK_HH
#define JETSTREAM_DOMAINS_IO_AUDIO_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct Audio : public Block::Config {
    std::string deviceName = "Default";
    F32 inSampleRate = 48e3;
    F32 outSampleRate = 48e3;
    F32 volume = 1.0f;

    JST_BLOCK_TYPE(audio);
    JST_BLOCK_DOMAIN("IO");
    JST_BLOCK_PARAMS(deviceName, inSampleRate, outSampleRate, volume);
    JST_BLOCK_DESCRIPTION(
        "Audio",
        "Audio playback device interface.",
        "# Audio\n"
        "The Audio block provides an interface to play audio through the system's "
        "playback devices. It resamples the input signal to match the output "
        "device sample rate when the rates differ.\n\n"

        "## Arguments\n"
        "- **Device**: Name of the audio playback device to use.\n"
        "- **Sample Rate**: Sample rate of the input signal in kHz.\n"
        "- **Output Sample Rate**: Sample rate for the audio device in kHz.\n"
        "- **Volume**: Volume multiplier (0.0 to 5.0). Values above 1.0 amplify the audio.\n\n"

        "## Useful For\n"
        "- Playing demodulated audio from radio signals.\n"
        "- Monitoring audio in real-time.\n"
        "- Audio output in signal processing pipelines.\n\n"

        "## Examples\n"
        "- Play demodulated FM audio:\n"
        "  Config: Device=Default, Sample Rate=48 kHz\n"
        "  Input: F32[4096] -> Audio playback.\n\n"

        "## Implementation\n"
        "Input Buffer -> Resampler -> Circular Buffer -> Audio Device\n"
        "1. Receives input samples at the specified input sample rate.\n"
        "2. Resamples to the output sample rate if different.\n"
        "3. Buffers samples in a circular buffer for the audio callback.\n"
        "4. Audio device callback pulls samples for playback."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_IO_AUDIO_BLOCK_HH
