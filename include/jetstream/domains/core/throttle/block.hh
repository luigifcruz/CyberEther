#ifndef JETSTREAM_DOMAINS_CORE_THROTTLE_BLOCK_HH
#define JETSTREAM_DOMAINS_CORE_THROTTLE_BLOCK_HH

#include "jetstream/block.hh"
#include "jetstream/types.hh"

namespace Jetstream::Blocks {

struct Throttle : public Block::Config {
    U64 intervalMs = 100;

    JST_BLOCK_TYPE(throttle);
    JST_BLOCK_DOMAIN("Core");
    JST_BLOCK_PARAMS(intervalMs);
    JST_BLOCK_DESCRIPTION(
        "Throttle",
        "Limits data flow rate by introducing time delays.",
        "# Throttle\n"
        "\n"
        "Passes each buffer through unchanged, waiting first until Interval has passed "
        "since the previous one. It usually follows a Signal Generator or File Reader, "
        "which otherwise run as fast as the computer allows.\n"
        "\n"
        "- **The whole flowgraph waits.** While it sleeps, every block pauses, "
        "including other chains.\n"
        "- **Passes are at least Interval apart.** They usually come slightly later, "
        "and a slower chain sets its own pace.\n"
        "- **Real time needs a matching buffer.** Size the source buffer to last "
        "Interval at its sample rate.\n"
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
        "| **Interval** | 100 ms | 1 ms or more, whole milliseconds | Shortest time "
        "between two buffers. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "| Goal | Settings | Input | Output |\n"
        "|---|---|---|---|\n"
        "| Ten updates per second | Defaults | 8,192 samples from a Signal Generator | "
        "The same 8,192 samples, about every 100 ms |\n"
        "| Test tone in real time | Interval 8 | 8,000 samples at 1 MS/s | The same "
        "8,000 samples, about every 8 ms |\n"
        "| Step through a recording | Interval 1,000 | 8,192 samples per batch from a "
        "File Reader | The same 8,192 samples, about once per second |\n"
        "| Chain slower than Interval | Interval 10 | Buffers whose chain takes 50 ms |"
        " Each buffer as it comes, with no added wait |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The block reports that Interval cannot be zero | Interval is 0 | Set "
        "Interval to 1 ms or more. |\n"
        "| Every view updates slowly, not only this chain | The block pauses the whole "
        "flowgraph while it waits | Shorten Interval, or remove the block when a radio "
        "already sets the pace. |\n"
        "| The radio's Buffer Loss climbs | The radio keeps sending while the flowgraph"
        " waits | Remove the Throttle, since a radio paces itself. |\n"
        "| Audio from a test source gaps or skips | The buffer does not last Interval, "
        "or passes run slightly late | Size the buffer to last Interval, such as 8,000 "
        "samples at 1 MS/s with Interval 8. |\n"
        "| The block reports that it expects a contiguous tensor | The input is "
        "strided, such as a Slice with Contiguous off | Turn on Contiguous in the "
        "Slice. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "The block sleeps on the compute thread until Interval has passed since its "
        "last pass, then hands on the input buffer without a copy. The first buffer, "
        "and the first one after Interval changes, passes at once.\n"
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
