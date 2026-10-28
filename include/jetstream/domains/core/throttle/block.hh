#ifndef JETSTREAM_DOMAINS_CORE_THROTTLE_BLOCK_HH
#define JETSTREAM_DOMAINS_CORE_THROTTLE_BLOCK_HH

#include "jetstream/block.hh"
#include "jetstream/types.hh"

namespace Jetstream::Blocks {

struct Throttle : public Block::Config {
    std::string pacing = "interval";
    F32 intervalMs = 100.0f;

    JST_BLOCK_TYPE(throttle);
    JST_BLOCK_DOMAIN("Core");
    JST_BLOCK_PARAMS(pacing, intervalMs);
    JST_BLOCK_DESCRIPTION(
        "Throttle",
        "Limits data flow rate by introducing time delays.",
        "# Throttle\n"
        "\n"
        "Passes each buffer through unchanged, holding it until a fixed Interval or the"
        " time the previous buffer lasts at its `sampleRate` has passed. It usually follows a Signal Generator or File Reader, which "
        "otherwise run as fast as the computer allows.\n"
        "\n"
        "- **Passes average one wait apart.** A late pass shortens the next wait, and a"
        " slower chain sets its own pace.\n"
        "- **Sample Rate pacing plays in real time.** Each buffer waits as long as it "
        "lasts at the input `sampleRate`.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Input** (in) | Any | Any shape. The buffer must be contiguous and on the "
        "CPU. |\n"
        "| **Output** (out) | Same as the input | The input buffer itself, with every "
        "axis and attribute kept. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **Pacing** | Interval | Interval, Sample Rate | Interval waits a fixed time. "
        "Sample Rate waits as long as each buffer lasts at the input `sampleRate`, "
        "counting samples per channel. |\n"
        "| **Interval** | 100 ms | 0.001 ms to 86,400,000 ms | Average time between two"
        " buffers, down to fractions of a millisecond. Hidden for Sample Rate. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "| Goal | Settings | Input | Output |\n"
        "|---|---|---|---|\n"
        "| Ten updates per second | Defaults | 8,192 samples from a Signal Generator | "
        "The same 8,192 samples, about every 100 ms |\n"
        "| Play a recording in real time | Pacing Sample Rate | 8,192 samples at 2 MS/s "
        "from a File Reader | The same 8,192 samples, about every 4.1 ms |\n"
        "| Low latency test tone | Interval 0.5 ms | 500 samples at 1 MS/s | The same "
        "500 samples, every 0.5 ms on average |\n"
        "| Step through a recording | Interval 1,000 ms | 8,192 samples per batch from "
        "a File Reader | The same 8,192 samples, about once per second |\n"
        "| Chain slower than Interval | Interval 10 ms | Buffers whose chain takes 50 "
        "ms | Each buffer as it comes, with no added wait |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The block reports that Interval must be between 0.001 ms and one day | "
        "Interval is below 0.001 ms or longer than a day | Set Interval between 0.001 "
        "ms and 86,400,000 ms. |\n"
        "| The radio's Buffer Loss climbs | The radio keeps sending while the chain "
        "waits | Remove the Throttle, since a radio paces itself. |\n"
        "| Audio from a test source gaps or skips | The buffer does not last Interval, "
        "or passes run slightly late | Size the buffer to last Interval, such as 8,000 "
        "samples at 1 MS/s with Interval 8 ms. |\n"
        "| Sample Rate pacing passes buffers at full speed | The input has no "
        "`sampleRate`, and the log warns about it | Use Interval pacing, or set the "
        "sample rate on the source. |\n"
        "| The block reports that it expects a contiguous tensor | The input is "
        "strided, such as a Slice with Contiguous off | Turn on Contiguous in the "
        "Slice. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "The block holds the buffer and tells the scheduler when the next pass is due, "
        "then hands on the input buffer without a copy. Each pass is due one wait after"
        " the previous one was due, unless that one ran more than one wait late, which "
        "restarts the schedule. The first buffer, and the first one after a setting "
        "changes, passes at once.\n"
        "\n"
        "## See Also\n"
        "\n"
        "- **Signal Generator** sends buffers as fast as the chain runs.\n"
        "- **File Reader** reads a new batch every cycle.\n"
        "- **Audio** needs a source paced in real time."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_CORE_THROTTLE_BLOCK_HH
