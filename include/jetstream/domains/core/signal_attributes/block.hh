#ifndef JETSTREAM_DOMAINS_CORE_SIGNAL_ATTRIBUTES_BLOCK_HH
#define JETSTREAM_DOMAINS_CORE_SIGNAL_ATTRIBUTES_BLOCK_HH

#include "jetstream/block.hh"
#include "jetstream/types.hh"

namespace Jetstream::Blocks {

struct SignalAttributes : public Block::Config {
    F32 sampleRate = 0.0f;
    F32 frequency = 0.0f;

    JST_BLOCK_TYPE(signal_attributes);
    JST_BLOCK_DOMAIN("Core");
    JST_BLOCK_PARAMS(sampleRate, frequency);
    JST_BLOCK_DESCRIPTION(
        "Signal Attributes",
        "Sets signal attributes.",
        "# Signal Attributes\n"
        "\n"
        "Sets the sample rate and center frequency that later blocks read, without "
        "touching the samples. It usually follows a source that sends none, such as a "
        "File Reader playing Raw I/Q.\n"
        "\n"
        "- **Zero keeps the input value.** Only a setting other than zero replaces what"
        " the source sent.\n"
        "- **Every later block sees the new values.** Views, filters, and Throttle read"
        " `sampleRate` and `frequency` from here.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Input** (in) | Any type | Any shape. Strided input is accepted. |\n"
        "| **Output** (out) | Same as the input | Same shape and data, with every axis "
        "and attribute kept. Sample Rate and Frequency replace `sampleRate` and "
        "`frequency` when not zero. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **Sample Rate** | 0 MHz | 0 MHz or more | Value for `sampleRate`. Zero keeps"
        " the input value. |\n"
        "| **Frequency** | 0 MHz | Any | Value for `frequency`. Zero keeps the input "
        "value, so this block cannot set 0 MHz. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "| Goal | Settings | Input | Output |\n"
        "|---|---|---|---|\n"
        "| Tag a raw capture | Sample Rate 2 MHz, Frequency 96.9 MHz | 8,192 complex "
        "samples with no attributes | The same samples at 2 MS/s, centered on 96.9 MHz "
        "|\n"
        "| Keep the source rate | Frequency 433.92 MHz | 8,192 samples at 2 MS/s | The "
        "same samples at 2 MS/s, centered on 433.92 MHz |\n"
        "| Fix a wrong rate | Sample Rate 1 MHz | 8,192 samples tagged at 2 MS/s | The "
        "same samples tagged at 1 MS/s |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The block reports that the sample rate must be zero or positive | Sample "
        "Rate is negative | Set Sample Rate to the capture rate, or to 0 to keep the "
        "input value. |\n"
        "| Views still show the source frequency | Frequency is 0, which keeps the "
        "input value | Set Frequency to the band center. |\n"
        "| Playback runs at the wrong speed | Sample Rate does not match how the signal"
        " was captured | Set Sample Rate to the capture rate. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "The output shares the input buffer and follows its attributes as they change, "
        "such as on a retune, while the values set here stay fixed.\n"
        "\n"
        "## See Also\n"
        "\n"
        "- **Signal Axes** sets which dimensions hold samples and channels.\n"
        "- **Throttle** paces playback by `sampleRate`.\n"
        "- **File Reader** plays Raw I/Q files that carry no attributes."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_CORE_SIGNAL_ATTRIBUTES_BLOCK_HH
