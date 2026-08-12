#ifndef JETSTREAM_DOMAINS_CORE_RANGE_BLOCK_HH
#define JETSTREAM_DOMAINS_CORE_RANGE_BLOCK_HH

#include "jetstream/block.hh"
#include "jetstream/types.hh"

namespace Jetstream::Blocks {

struct Range : public Block::Config {
    F32 min = -1.0f;
    F32 max = +1.0f;

    JST_BLOCK_TYPE(range);
    JST_BLOCK_DOMAIN("Core");
    JST_BLOCK_PARAMS(min, max);
    JST_BLOCK_DESCRIPTION(
        "Range",
        "Compresses input into a normalized range.",
        "# Range\n"
        "The Range block normalizes input data with a smooth soft knee. Values outside the selected minimum "
        "and maximum are compressed toward 0 and 1 instead of being clipped or extending beyond the display.\n\n"

        "## Arguments\n"
        "- **Min**: The lower soft-knee value of the input range.\n"
        "- **Max**: The upper soft-knee value of the input range.\n"
        "The bounds are ordered automatically if they cross. Equal bounds produce a constant output of 0.5.\n\n"

        "## Useful For\n"
        "- Normalizing signal amplitudes for visualization.\n"
        "- Compressing decibel values into display bounds.\n"
        "- Preparing data for display on fixed-range indicators.\n\n"

        "## Examples\n"
        "- Normalize dB levels:\n"
        "  Config: Min=-100, Max=0\n"
        "  Input: F32[1024] around [-100, 0] dB -> Output: F32[1024] softly compressed into [0, 1]\n\n"

        "## Implementation\n"
        "Input -> Normalize -> Apply tanh soft knee -> Output\n"
        "After ordering the bounds, the transform is: normalized = (input - min) / (max - min), "
        "output = 0.5 + 0.5 * tanh(4 * (normalized - 0.5)).";
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_CORE_RANGE_BLOCK_HH
