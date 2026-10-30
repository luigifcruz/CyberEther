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
        "Plays audio through a sound device.",
        "# Audio\n"
        "\n"
        "Plays mono or stereo samples on a sound device. It usually ends a radio chain,"
        " right after an FM or AM Demodulator.\n"
        "\n"
        "- **Sample Rate is not read from the input.** Set it to match the block "
        "upstream.\n"
        "- **Playback runs at 48 kHz.** Other input rates are resampled to it.\n"
        "- **The source must run in real time.** A File Reader or Signal Generator "
        "needs a Throttle to keep pace.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Input** (in) | F32 | Needs a `sampleAxis`, implied for one-dimensional "
        "input. Batches play in order, and a `channelAxis` of 2 is stereo, left then "
        "right. The buffer must be contiguous. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **Device** | Default | Default and the playback devices found | Sound "
        "device. Default follows the system choice. |\n"
        "| **Sample Rate** | 48 kHz | Above 0 | Rate of the input. It is not read from "
        "the input, so it must match the block upstream. |\n"
        "| **Volume** | 1 | 0 or more, slider up to 5 | Linear gain. Above 1 amplifies,"
        " and the output clips at full scale. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "| Goal | Settings | Input | Output |\n"
        "|---|---|---|---|\n"
        "| Narrowband FM | Sample Rate 200 kHz | 819 real samples at 200 kS/s | 4.1 ms "
        "of mono audio per buffer |\n"
        "| Stereo broadcast | Sample Rate 200 kHz | 819 samples x 2 channels at 200 "
        "kS/s | 4.1 ms of stereo audio per buffer |\n"
        "| Batched stereo | Sample Rate 240 kHz | 8 batches x 1,000 samples x 2 "
        "channels at 240 kS/s | 33.3 ms of stereo audio per buffer, batches in order |\n"
        "| No resampling | Sample Rate 48 kHz | 4,800 samples at 48 kS/s | 100 ms of "
        "mono audio per buffer, played as is |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| Audio plays too slow and low, or too fast and high | Sample Rate does not "
        "match the input | Set Sample Rate to the rate of the block upstream. |\n"
        "| Audio stutters or skips | The source delivers samples faster or slower than "
        "real time | Feed it from a radio, or add a Throttle after a file or test "
        "source. |\n"
        "| The block reports that the input must contain one or two audio channels | "
        "The `channelAxis` holds more than 2, such as audio from a multi-head Filter | "
        "Pick one head with a Slice block. |\n"
        "| The block reports that the input buffer must be F32 | The input is complex, "
        "such as raw radio samples | Demodulate it first. |\n"
        "| The log warns that the device is not found, using default | The saved Device"
        " is not connected | Pick a device again. Sound plays on Default meanwhile. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "Each buffer is put in playback order and resampled to the device rate by a "
        "linear resampler with a low-pass filter. The sound device pulls samples from a"
        " ring buffer on its own clock, playing silence when the buffer runs short. "
        "When the buffer fills, the oldest audio is overwritten.\n"
        "\n"
        "## See Also\n"
        "\n"
        "- Mono or stereo audio comes from the **FM Demodulator**.\n"
        "- Amplitude-modulated audio comes from the **AM Demodulator**.\n"
        "- **Throttle** paces file and test sources.\n"
        "- **Slice** picks one head from multi-head audio."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_IO_AUDIO_BLOCK_HH
