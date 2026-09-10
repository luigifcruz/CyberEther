#ifndef JETSTREAM_DOMAINS_DSP_DECIMATOR_BLOCK_HH
#define JETSTREAM_DOMAINS_DSP_DECIMATOR_BLOCK_HH

#include <string>

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct Decimator : public Block::Config {
    U64 ratio = 4;
    std::string method = "sum";

    JST_BLOCK_TYPE(decimator);
    JST_BLOCK_DOMAIN("DSP");
    JST_BLOCK_PARAMS(ratio, method);
    JST_BLOCK_DESCRIPTION(
        "Decimator",
        "Decimates a signal along its sample axis.",
        "# Decimator\n"
        "The Decimator block reduces the sample count by reducing each group "
        "of consecutive samples to one output sample. The method selects the "
        "first sample, sums the group, or averages it. The input must "
        "identify its sample dimension with `sampleAxis`. Optional "
        "`batchAxis` and `channelAxis` roles are preserved on the output. "
        "All methods divide `sampleRate` metadata by the ratio when present.\n\n"

        "## Arguments\n"
        "- **Ratio**: Samples per group. Must be positive "
        "and divide the sample dimension evenly.\n"
        "- **Method**: Subsample (`subsample`), "
        "Sum (`sum`, default), or Average (`average`).\n\n"

        "## Methods\n"
        "- **Subsample**: Keep samples at indices 0, Ratio, 2*Ratio, and so on.\n"
        "- **Sum**: Add each group's samples. A constant "
        "input's amplitude is multiplied by the ratio.\n"
        "- **Average**: Compute each group's arithmetic "
        "mean, preserving a constant input's amplitude.\n"
        "Subsample performs no filtering. Sum and Average apply a boxcar filter "
        "with different gain. Complex samples are combined component-wise.\n\n"

        "## Useful For\n"
        "- Implementing decimation filters for signal processing.\n"
        "- Downsampling data by a fixed ratio.\n"
        "- Aggregating sensor data from multiple sources.\n\n"

        "## Examples\n"
        "- Time-domain decimation:\n"
        "  Config: Ratio=4\n"
        "  Input: CF32[8192] -> Output: CF32[2048]\n"
        "- With Ratio=4, input [1, 2, 3, 4, 5, 6, 7, 8] produces:\n"
        "  Subsample: [1, 5]; Sum: [10, 26]; Average: [2.5, 6.5].\n\n"

        "## Implementation\n"
        "Subsample uses a strided Slice. Sum reshapes the sample axis into "
        "groups, adds along the group axis, and squeezes that axis. Average "
        "scales samples by the reciprocal ratio before summing. Duplicate "
        "ensures proper output buffering and host accessibility."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_DSP_DECIMATOR_BLOCK_HH
