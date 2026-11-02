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
        "\n"
        "Lowers the sample rate by a whole-number Ratio, turning each group of Ratio "
        "samples into one. It usually follows a Filter that band-limits the signal, and"
        " feeds blocks that run at the lower rate.\n"
        "\n"
        "- **The length must divide evenly.** Every buffer's sample axis must be a "
        "multiple of Ratio.\n"
        "- **The sample rate drops by Ratio.** The `sampleRate` attribute is divided by"
        " Ratio.\n"
        "- **Nothing filters the aliases.** Band-limit the signal first when content "
        "sits above the new rate.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Input** (in) | F32 or CF32. Subsample takes any type. | Needs a "
        "`sampleAxis`, implied for one-dimensional input. A `batchAxis` and a "
        "`channelAxis` are optional. |\n"
        "| **Output** (out) | Same as the input | The input shape with the sample axis "
        "divided by Ratio. Every axis keeps its role. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **Ratio** | 4 | 1 or more, dividing the sample axis length | Input samples "
        "per output sample. |\n"
        "| **Method** | Sum | Subsample, Sum, Average | Subsample keeps the first "
        "sample of each group. Sum adds the group. Average takes its mean. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "All rows assume a 2 MS/s source.\n"
        "\n"
        "| Goal | Settings | Input | Output |\n"
        "|---|---|---|---|\n"
        "| Quarter rate, same level | Average | 8,192 samples | 2,048 samples at 500 "
        "kS/s |\n"
        "| Quarter rate with gain | Defaults | 8,192 samples | 2,048 samples at 500 "
        "kS/s, 4 times louder |\n"
        "| Thin an oversampled signal | Subsample, Ratio 8 | 8,192 samples | 1,024 "
        "samples at 250 kS/s |\n"
        "| Batched rows | Average, Ratio 10 | 8 batches x 1,000 samples | 8 batches x "
        "100 samples at 200 kS/s |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The block reports that the axis size is not divisible by the ratio | The "
        "sample axis length is not a multiple of Ratio | Pick a Ratio that divides the "
        "length, or change the buffer size upstream. |\n"
        "| The level rises by Ratio | Method is Sum, the default | Switch Method to "
        "Average. |\n"
        "| Signals appear at the wrong frequency | Content above half the new rate "
        "aliases | Band-limit first with a Filter, or lower Ratio. |\n"
        "| Sum or Average reports an unsupported data type | The input is not F32 or "
        "CF32 | Convert it with a Cast block, or use Subsample. |\n"
        "| Sum reports that it expects a contiguous tensor | The input is strided, such"
        " as a Slice with Contiguous off | Turn on Contiguous in the Slice, or use "
        "Average. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "Subsample keeps the first sample of each group through a strided view. Sum and"
        " Average split the sample axis into groups and add each one, with Average also"
        " dividing by the group size. The model below covers one buffer.\n"
        "\n"
        "```python\n"
        "def decimate(x, ratio=4, method=\"sum\"):\n"
        "    if method == \"subsample\":\n"
        "        return x[::ratio]\n"
        "    groups = x.reshape(-1, ratio)\n"
        "    if method == \"sum\":\n"
        "        return groups.sum(axis=1)\n"
        "    return groups.mean(axis=1)\n"
        "```\n"
        "\n"
        "## See Also\n"
        "\n"
        "- **Filter** band-limits and decimates in one step.\n"
        "- **Rational Resampler** handles non-integer rate changes.\n"
        "- **Cast** converts integer input for Sum and Average."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_DSP_DECIMATOR_BLOCK_HH
