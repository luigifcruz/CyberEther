#ifndef JETSTREAM_DOMAINS_DSP_SQUELCH_BLOCK_HH
#define JETSTREAM_DOMAINS_DSP_SQUELCH_BLOCK_HH

#include "jetstream/block.hh"

// TODO: Add smoothing to the threshold/amplitude state.
// TODO: Add hysteresis to reduce chatter around the threshold.
// TODO: Add a "Set Threshold" button in the UI.

namespace Jetstream::Blocks {

struct Squelch : public Block::Config {
    F32 threshold = 0.1f;

    JST_BLOCK_TYPE(squelch);
    JST_BLOCK_PARAMS(threshold);
    JST_BLOCK_DESCRIPTION(
        "Squelch",
        "Passes input only when signal strength is above a threshold.",
        "# Squelch\n"
        "The Squelch block monitors input signal strength and forwards the input only when the "
        "signal is above a configurable threshold. When the signal remains below threshold, "
        "downstream processing is skipped for that buffer.\n\n"

        "## Arguments\n"
        "- **Threshold**: The signal level threshold for squelch activation (linear amplitude).\n"
        "\n"

        "## Useful For\n"
        "- Muting noise in radio receivers when no carrier is present.\n"
        "- Skipping downstream DSP when no useful signal is present.\n"
        "- Gating signals based on amplitude for cleaner processing.\n"
        "- Reducing power consumption by disabling downstream processing.\n\n"

        "## Examples\n"
        "- Simple squelch with 0.1 amplitude threshold:\n"
        "  Config: Threshold=0.1\n"
        "  Input: CF32[1024] with varying amplitude -> Output: CF32[1024] forwarded only when open\n"
        "- Tighter squelch at 0.5 amplitude:\n"
        "  Config: Threshold=0.5\n"
        "  Input: F32[2048] -> Output: F32[2048] forwarded only when the signal exceeds 0.5\n\n"

        "## Implementation\n"
        "Input -> Level Check -> Pass or Skip\n"
        "1. Measures sample amplitude against the threshold.\n"
        "2. Tracks the peak amplitude of the latest buffer.\n"
        "3. Passes the input tensor through unchanged when open.\n"
        "4. Skips downstream work when the buffer remains closed."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_DSP_SQUELCH_BLOCK_HH
