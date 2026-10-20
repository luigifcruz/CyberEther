#ifndef JETSTREAM_DOMAINS_CORE_REMOVE_INDICES_BLOCK_HH
#define JETSTREAM_DOMAINS_CORE_REMOVE_INDICES_BLOCK_HH

#include <vector>

#include "jetstream/block.hh"
#include "jetstream/types.hh"

namespace Jetstream::Blocks {

struct RemoveIndices : public Block::Config {
    I64 axis = -1;
    std::vector<U64> indices = {};

    JST_BLOCK_TYPE(remove_indices);
    JST_BLOCK_DOMAIN("Core");
    JST_BLOCK_PARAMS(axis, indices);
    JST_BLOCK_DESCRIPTION(
        "Remove Indices",
        "Removes entries by index.",
        "# Remove Indices\n"
        "\n"
        "Deletes the entries at chosen positions along one axis, keeping the rest in "
        "their original order. It usually follows a source with many antennas or "
        "channels, to drop the ones that are faulty or unused.\n"
        "\n"
        "- **Later entries move up.** After removing index 2, the entry that was 3 sits"
        " at 2.\n"
        "- **The default removes nothing.** Empty Indices outputs a contiguous copy of "
        "the input.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Input** (in) | Any type | Any shape. Strided input is accepted. |\n"
        "| **Output** (out) | Same as the input | The input shape with Axis shortened "
        "by the distinct Indices, in a new contiguous buffer. Every role and attribute "
        "is kept. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **Axis** | -1 | From -N to N - 1, for N input axes | Axis to remove entries "
        "from, counted from 0 at the outermost. Negative values count from the end. |\n"
        "| **Indices** | [] | Positions below the length of Axis, leaving at least one "
        "| Positions to remove, counted from 0 in the input. Repeats count once. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "| Goal | Settings | Input | Output |\n"
        "|---|---|---|---|\n"
        "| Drop three bad antennas | Axis 0, Indices [2, 7, 19] | 28 batches x 8,192 "
        "samples | 25 batches x 8,192 samples |\n"
        "| Drop the first and last channel | Axis 0, Indices [0, 7] | 8 channels x "
        "1,024 samples | 6 channels x 1,024 samples |\n"
        "| Repeated entries | Axis 0, Indices [1, 1, 3] | 4 channels x 1,024 samples | "
        "2 channels x 1,024 samples |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The block reports that an index is out of range for the axis | An entry in "
        "Indices is not below the length of Axis | Remove that entry, or point Axis at "
        "the right dimension. |\n"
        "| The block reports that it cannot remove every entry along an axis | Indices "
        "covers every position of Axis | Keep at least one position out of Indices. |\n"
        "| The block reports that the axis is out of range | Axis is outside -N to N - "
        "1 for N input axes | Count axes from 0, or from -1 at the end. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "The block works out once which entries survive, then copies them into the "
        "output every buffer, joining neighbors into single copies. Strided input is "
        "read one element at a time. The model below covers one buffer.\n"
        "\n"
        "```python\n"
        "import numpy as np\n"
        "\n"
        "def remove_indices(x, axis=-1, indices=()):\n"
        "    removed = sorted(set(indices))\n"
        "    if len(removed) == x.shape[axis]:\n"
        "        raise ValueError(\"Cannot remove every entry along an axis.\")\n"
        "    y = np.delete(x, np.array(removed, dtype=int), axis=axis)\n"
        "    return np.ascontiguousarray(y)\n"
        "```\n"
        "\n"
        "## See Also\n"
        "\n"
        "- **Slice** keeps a range or a single index instead.\n"
        "- **Unpad** removes entries from the end of an axis.\n"
        "- **Permutation** reorders axes without removing any."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_CORE_REMOVE_INDICES_BLOCK_HH
