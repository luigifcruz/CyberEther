#ifndef JETSTREAM_DOMAINS_CORE_ARITHMETIC_BLOCK_HH
#define JETSTREAM_DOMAINS_CORE_ARITHMETIC_BLOCK_HH

#include <string>

#include "jetstream/block.hh"
#include "jetstream/types.hh"

namespace Jetstream::Blocks {

struct Arithmetic : public Block::Config {
    std::string operation = "add";
    I64 axis = -1;
    bool squeeze = false;

    JST_BLOCK_TYPE(arithmetic);
    JST_BLOCK_DOMAIN("Core");
    JST_BLOCK_PARAMS(operation, axis, squeeze);
    JST_BLOCK_DESCRIPTION(
        "Arithmetic",
        "Reduces a tensor along an axis using an arithmetic operation.",
        "# Arithmetic\n"
        "\n"
        "Combines the values along one axis of a tensor into a single value, such as "
        "the sum of each batch. It usually condenses batches or heads before a plot, or"
        " merges the heads of a Filter into one stream.\n"
        "\n"
        "- **Only Add combines values correctly.** The other operations start from 0, "
        "so their results are wrong.\n"
        "- **The sample rate stays the same.** Reducing the sample axis still reports "
        "the source `sampleRate`.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Input** (in) | F32 or CF32 | Any shape with at least one axis. Strided "
        "input is accepted. |\n"
        "| **Output** (out) | Same as the input | The input shape with Axis set to size"
        " 1, or removed with Squeeze on. Every axis keeps its role, except a squeezed "
        "one, and every attribute is kept. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **Operation** | Add | Add, Subtract, Multiply, Divide | Add sums the values."
        " The other operations give wrong results, as Troubleshooting shows. |\n"
        "| **Axis** | -1 | From -N to N - 1, for N input axes | Axis to combine, "
        "counted from 0 at the outermost. Negative values count from the end. |\n"
        "| **Squeeze** | Off | On, Off | Removes the combined axis and its role from "
        "the output. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "| Goal | Settings | Input | Output |\n"
        "|---|---|---|---|\n"
        "| Sum each batch | Defaults | 8 batches x 1,024 samples | 8 batches x 1 sample"
        " |\n"
        "| One sum per batch | Squeeze on | 8 batches x 1,024 samples | 8 batches, with"
        " no `sampleAxis` |\n"
        "| Merge Filter heads | Axis 0, Squeeze on | 3 heads x 819 samples | 819 "
        "samples |\n"
        "| Total of a buffer | Defaults | 8,192 samples | 1 sample |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| Subtract returns the negated sum, and Multiply or Divide returns zeros or "
        "NaN | Every operation starts from 0, a bug in the block | Use Add, or compute "
        "the reduction in a Python block. |\n"
        "| The block reports that the axis is out of range for the input rank | Axis is"
        " not an axis of the input | Pick an axis within the input, such as -1 for the "
        "last. |\n"
        "| The block reports an unsupported data type | The input is not F32 or CF32 | "
        "Convert it with a Cast block. |\n"
        "| A block downstream reports that `sampleAxis` and `batchAxis` cannot use axis"
        " 0 | Squeezing the sample axis of batched input leaves one axis marked as "
        "batches | Relabel it with a Signal Axes block, such as [S]. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "The block starts from zeros with the combined axis at size one, then folds "
        "each input value into its slot with the chosen operation. The model below "
        "covers one buffer.\n"
        "\n"
        "```python\n"
        "import numpy as np\n"
        "\n"
        "OPS = {\"add\": np.add, \"sub\": np.subtract, \"mul\": np.multiply, \"div\": np.divide}\n"
        "\n"
        "def arithmetic(x, operation=\"add\", axis=-1, squeeze=False):\n"
        "    out = np.zeros_like(np.take(x, [0], axis=axis))\n"
        "    for i in range(x.shape[axis]):\n"
        "        out = OPS[operation](out, np.take(x, [i], axis=axis))\n"
        "    return np.squeeze(out, axis=axis) if squeeze else out\n"
        "```\n"
        "\n"
        "## See Also\n"
        "\n"
        "- **Add** sums two tensors element by element instead.\n"
        "- **Python** computes other reductions.\n"
        "- **Signal Axes** relabels the axes after a squeeze.\n"
        "- **Squeeze Dims** removes a size-one axis afterwards."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_CORE_ARITHMETIC_BLOCK_HH
