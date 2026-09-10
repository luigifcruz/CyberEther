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
        "Normalizes input using an affine range mapping.",
        "# Range\n"
        "The Range block maps the selected minimum and maximum to 0 and 1. "
        "Values outside the selected range extend below 0 or above 1. This affine "
        "mapping preserves averages, including averages of decibel values.\n\n"

        "## Arguments\n"
        "- **Min**: The input value mapped to 0.\n"
        "- **Max**: The input value mapped to 1.\n"
        "The bounds are ordered automatically if they cross. Equal bounds produce a constant output of 0.5.\n\n"

        "## Useful For\n"
        "- Normalizing signal amplitudes for visualization.\n"
        "- Scaling decibel values for log-domain averaging.\n"
        "- Preparing data for display on fixed-range indicators.\n\n"

        "## Examples\n"
        "- Normalize dB levels:\n"
        "  Config: Min=-100, Max=0\n"
        "  Input: F32[1024] in [-100, 0] dB -> Output: F32[1024] in [0, 1]\n\n"

        "## Implementation\n"
        "Input -> Affine Normalization -> Output\n"
        "After ordering the bounds, the transform is: output = (input - min) / (max - min).";
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_CORE_RANGE_BLOCK_HH
