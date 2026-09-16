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
        "The Remove Indices block removes entire slices at a list of indices "
        "along one axis, like NumPy's `delete` with an explicit axis. Remaining "
        "entries retain their original order. The output is a contiguous copy "
        "with the same rank and data type as the input.\n\n"

        "## Arguments\n"
        "- **Axis**: The dimension to remove entries from. Negative axes count "
        "from the end.\n"
        "- **Indices**: Zero-based indices in the original input. Duplicates "
        "count once, and an empty list keeps every entry. Out-of-range indices "
        "and removal of every entry along the axis are rejected.\n\n"

        "## Useful For\n"
        "- Removing bad antennas from telescope packets.\n"
        "- Excluding selected channels, time samples, or polarizations.\n\n"

        "## Examples\n"
        "- Remove three antennas from an [A, C, T, P] tensor:\n"
        "  Config: Axis=0, Indices=[2, 7, 19]\n"
        "  Input: CF32[28, 192, 8192, 2] -> Output: CF32[25, 192, 8192, 2]\n"
        "- Remove the first and last columns:\n"
        "  Config: Axis=-1, Indices=[0, 7]\n"
        "  Input: F32[4, 8] -> Output: F32[4, 6]\n\n"

        "## Implementation\n"
        "Input -> Remove Indices -> Output\n"
        "The selection is prepared once and reused for each input packet. "
        "Native CPU and CUDA implementations support contiguous and strided "
        "inputs. CUDA processing keeps the output on the GPU. "
        "Configuration changes recreate the module and its output."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_CORE_REMOVE_INDICES_BLOCK_HH
