#ifndef JETSTREAM_DOMAINS_CORE_PERMUTATION_BLOCK_HH
#define JETSTREAM_DOMAINS_CORE_PERMUTATION_BLOCK_HH

#include <vector>

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct Permutation : public Block::Config {
    std::vector<U64> permutation = {0};
    bool contiguous = false;

    JST_BLOCK_TYPE(permutation);
    JST_BLOCK_DOMAIN("Core");
    JST_BLOCK_PARAMS(permutation, contiguous);
    JST_BLOCK_DESCRIPTION(
        "Permutation",
        "Reorders tensor axes with a user-defined permutation.",
        "# Permutation\n"
        "\n"
        "Reorders the axes of a tensor, such as swapping rows and columns, without "
        "changing its values. It usually sits between two blocks that expect the same "
        "axes in a different order.\n"
        "\n"
        "- **The default only fits one-dimensional input.** Enter one index per input "
        "axis for anything larger.\n"
        "- **Axis roles move with their axes.** A `sampleAxis` moved to the end is "
        "still the `sampleAxis` there.\n"
        "- **The output is a view by default.** Turn on Contiguous for blocks that "
        "reject strided buffers.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Input** (in) | Any type | Any shape with as many axes as Permutation has "
        "entries. Strided input is accepted. |\n"
        "| **Output** (out) | Same as the input | The input axes in the order of "
        "Permutation. Every role and attribute is kept. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **Permutation** | [0] | Each input axis index once, from 0 | Input axis "
        "placed at each output position, outermost first. |\n"
        "| **Contiguous** | Off | On, Off | Copies the reordered view into a new "
        "contiguous buffer every cycle. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "| Goal | Settings | Input | Output |\n"
        "|---|---|---|---|\n"
        "| Swap batches and samples | [1, 0] | 8 batches x 1,024 samples | 1,024 "
        "samples x 8 batches, strided |\n"
        "| Move the first axis last | [1, 2, 0] | 8 x 64 x 1,024 values | 64 x 1,024 x "
        "8 values, strided |\n"
        "| Transpose into a new buffer | [1, 0], Contiguous on | 64 x 1,024 values | "
        "1,024 x 64 values in a new buffer |\n"
        "| One-dimensional input | Defaults | 8,192 samples | 8,192 samples, unchanged "
        "|\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The block reports that the input rank does not match the permutation size | "
        "Permutation has more or fewer entries than the input has axes, such as the "
        "default [0] on two-dimensional input | Enter one index per input axis, such as"
        " [1, 0]. |\n"
        "| The block reports that an axis is out of range or appears more than once | "
        "An entry is not below the number of entries, or repeats | Use each index from "
        "0 to the last axis exactly once. |\n"
        "| The block reports that the permutation cannot be empty | Permutation is [] |"
        " Enter one index per input axis. |\n"
        "| A block downstream reports that it expects a contiguous tensor | The output "
        "is a strided view of the input | Turn on Contiguous. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "The block only reorders the shape and strides that describe the input, so no "
        "values move and nothing is computed per buffer. Contiguous adds a copy into a "
        "new buffer on the same device. The model below covers one buffer.\n"
        "\n"
        "```python\n"
        "import numpy as np\n"
        "\n"
        "def permute(x, permutation=(0,), contiguous=False):\n"
        "    if sorted(permutation) != list(range(x.ndim)):\n"
        "        raise ValueError(\"Invalid permutation.\")\n"
        "    y = np.transpose(x, permutation)\n"
        "    return np.ascontiguousarray(y) if contiguous else y\n"
        "```\n"
        "\n"
        "## See Also\n"
        "\n"
        "- **Reshape** regroups values instead of reordering axes.\n"
        "- **Signal Axes** assigns roles to the reordered axes.\n"
        "- **Slice** keeps part of an axis."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_CORE_PERMUTATION_BLOCK_HH
