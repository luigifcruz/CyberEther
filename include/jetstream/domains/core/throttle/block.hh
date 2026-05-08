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
        "The Throttle block controls the rate of data flow through the flowgraph by enforcing "
        "a minimum time interval between data passes. It passes input data through unchanged "
        "but ensures that outputs are only produced at the specified rate.\n\n"

        "## Arguments\n"
        "- **Interval**: Minimum time between outputs in milliseconds.\n\n"

        "## Useful For\n"
        "- Rate limiting visualization updates.\n"
        "- Controlling CPU usage in processing pipelines.\n"
        "- Synchronizing data flow with external timing requirements.\n"
        "- Testing and debugging flowgraph timing behavior.\n\n"

        "## Examples\n"
        "- Limit updates to 10 Hz (100ms interval):\n"
        "  Input: Any tensor -> Output: Same tensor (at limited rate)\n\n"

        "## Implementation\n"
        "Input -> Throttle Module -> Output\n"
        "1. Checks elapsed time since last output.\n"
        "2. If interval has not elapsed, yields execution.\n"
        "3. If interval has elapsed, passes data through and updates timestamp."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_CORE_THROTTLE_BLOCK_HH
