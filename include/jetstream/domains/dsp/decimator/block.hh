#ifndef JETSTREAM_DOMAINS_DSP_DECIMATOR_BLOCK_HH
#define JETSTREAM_DOMAINS_DSP_DECIMATOR_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct Decimator : public Block::Config {
    U64 axis = 1;
    U64 ratio = 4;

    JST_BLOCK_TYPE(decimator);
    JST_BLOCK_PARAMS(axis, ratio);
    JST_BLOCK_DESCRIPTION(
        "Decimator",
        "Decimates a signal by summing along an axis.",
        "# Decimator\n"
        "The Decimator block reduces the size of a tensor along a specified axis "
        "by grouping consecutive elements into chunks of the given ratio and "
        "summing each chunk. This effectively downsamples the signal while "
        "preserving accumulated energy.\n\n"

        "## Arguments\n"
        "- **Axis**: The axis along which to decimate the input tensor.\n"
        "- **Ratio**: The decimation factor that determines chunk size.\n\n"

        "## Useful For\n"
        "- Implementing decimation filters for signal processing.\n"
        "- Downsampling data by a fixed ratio.\n"
        "- Aggregating sensor data from multiple sources.\n\n"

        "## Examples\n"
        "- Time-domain decimation:\n"
        "  Config: Axis=1, Ratio=4\n"
        "  Input: CF32[8192] -> Output: CF32[2048]\n\n"

        "## Implementation\n"
        "Input -> Reshape -> Add Axis -> Squeeze Axis -> Duplicate -> Output\n"
        "1. Reshape module separates the specified axis into ratio chunks.\n"
        "2. Arithmetic module sums all elements along the new ratio axis.\n"
        "3. Duplicate module ensures proper output buffering and host "
        "accessibility."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_DSP_DECIMATOR_BLOCK_HH
