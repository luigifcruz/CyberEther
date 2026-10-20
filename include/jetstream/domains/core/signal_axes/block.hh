#ifndef JETSTREAM_DOMAINS_CORE_SIGNAL_AXES_BLOCK_HH
#define JETSTREAM_DOMAINS_CORE_SIGNAL_AXES_BLOCK_HH

#include <string>

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct SignalAxes : public Block::Config {
    std::string axes;

    JST_BLOCK_TYPE(signal_axes);
    JST_BLOCK_DOMAIN("Core");
    JST_BLOCK_PARAMS(axes);
    JST_BLOCK_DESCRIPTION(
        "Signal Axes",
        "Assigns signal roles to tensor dimensions.",
        "# Signal Axes\n"
        "\n"
        "Sets which dimensions of a tensor hold batches, channels, and samples, without"
        " moving any data. It usually follows a Reshape or another block whose output "
        "has lost its roles.\n"
        "\n"
        "- **Roles you do not write are cleared.** Mark an axis with `*` to keep the "
        "role the input gave it.\n"
        "- **Blank Axes keeps every role.** The block then only checks that the input "
        "roles are valid.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Input** (in) | Any type | Any shape. Strided input is accepted. A "
        "one-dimensional input with no roles counts as samples. |\n"
        "| **Output** (out) | Same as the input | Same shape, values, and attributes, "
        "with `sampleAxis`, `batchAxis`, and `channelAxis` set by Axes. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **Axes** | Blank | Blank, or entries in brackets, see Reference | Role of "
        "each axis, outermost first. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "| Goal | Settings | Input | Output |\n"
        "|---|---|---|---|\n"
        "| Restore roles after a Reshape | Axes `[B, S]` | 8 x 1,024 values with no "
        "roles | 8 batches x 1,024 samples |\n"
        "| Channels of samples | Axes `[C, S]` | 4 x 8,192 values with no roles | 4 "
        "channels x 8,192 samples |\n"
        "| One value per channel for a plot | Axes `[B, C]` | 2 x 16 values | 2 batches"
        " x 16 channels, accepted by Lineplot |\n"
        "| Keep the batch, relabel the rest | Axes `[*, C]` | 2 batches x 8 samples | 2"
        " batches x 8 channels |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The block reports an invalid axes layout | The entries break a rule in "
        "Reference, such as a repeated role or more entries than dimensions | Fix the "
        "entries as in Reference. The log gives the exact cause. |\n"
        "| The block reports that a role is assigned to one axis and inherited from "
        "another | A `*` sits on an input axis whose role Axes also assigns elsewhere |"
        " Replace that `*` with `_`, or drop the other assignment. |\n"
        "| The block reports that the input signal axis metadata is invalid | The input"
        " roles are broken, and Axes is blank or keeps them with `*` | Write every role"
        " without `*`, which replaces the input roles. |\n"
        "| A block downstream reports a missing `sampleAxis` | Axes has no S, or the "
        "sample role was dropped without a `*` | Add S, or put `*` on the sample axis. "
        "|\n"
        "\n"
        "## Reference\n"
        "\n"
        "Entries are separated by commas and give the role of each axis from the "
        "outermost, and axes after the last entry get no role.\n"
        "\n"
        "| Entry | Meaning |\n"
        "|---|---|\n"
        "| **B** | Batch, sets `batchAxis`. |\n"
        "| **C** | Channel, sets `channelAxis`. |\n"
        "| **S** | Sample, sets `sampleAxis`. |\n"
        "| **_** | No role, so `[_]` alone clears every role. |\n"
        "| **\\*** | Keeps the role the input has on this axis, if any. |\n"
        "\n"
        "Each role appears once, and letters are uppercase.\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "The output is a view of the input memory with its own attributes, so no values"
        " move and nothing is computed per buffer.\n"
        "\n"
        "## See Also\n"
        "\n"
        "- **Reshape** regroups dimensions and drops their roles.\n"
        "- **Permutation** reorders dimensions and moves their roles with them.\n"
        "- **Python** declares output roles in its tensor specs.\n"
        "- **Lineplot** draws channel-only tensors from this block."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_CORE_SIGNAL_AXES_BLOCK_HH
