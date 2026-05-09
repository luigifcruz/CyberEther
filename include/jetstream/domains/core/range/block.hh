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
        "Scales input to a specified range.",
        "# Range\n"
        "The Range block normalizes input data by scaling and offsetting values to fit within a specified "
        "minimum and maximum range. This is useful for normalizing signal amplitudes or converting between "
        "different value domains.\n\n"

        "## Arguments\n"
        "- **Min**: The minimum value of the input range.\n"
        "- **Max**: The maximum value of the input range.\n\n"

        "## Useful For\n"
        "- Normalizing signal amplitudes for visualization.\n"
        "- Converting decibel values to linear scale within bounds.\n"
        "- Preparing data for display on fixed-range indicators.\n\n"

        "## Examples\n"
        "- Normalize dB levels:\n"
        "  Config: Min=-100, Max=0\n"
        "  Input: F32[1024] with range [-100, 0] dB -> Output: F32[1024] normalized to [0, 1]\n\n"

        "## Implementation\n"
        "Input -> Scale by 1/(max-min) -> Offset by -min/(max-min) -> Output\n"
        "The scaling coefficient is computed as: output = (input - min) / (max - min).";
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_CORE_RANGE_BLOCK_HH
